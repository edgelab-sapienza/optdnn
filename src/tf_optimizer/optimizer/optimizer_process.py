import os
import shutil
import tempfile
from typing import Optional

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.sparsity.keras import prunable_layer, prune_registry
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_policy import PruningPolicy

from tf_optimizer.dataset_manager import DatasetManager
from tf_optimizer.optimizer.optimization_param import ModelProblemInt, PruningPlan

"""
This class contains all the static code that will be spawned on dedicated processes for the optimization
"""


class MyThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs["val_accuracy"]
        if val_acc >= self.threshold:
            self.model.stop_training = True


class MyPolicy(tfmot.sparsity.keras.PruneForLatencyOnXNNPack):
    def allow_pruning(self, layer):
        return (
                isinstance(layer, prunable_layer.PrunableLayer)
                or hasattr(layer, "get_prunable_weights")
                or prune_registry.PruneRegistry.supports(layer)
        )

    def ensure_model_supports_pruning(self, model):
        return True

    def _lookup_layers(self, source_layers, stop_fn, next_fn):
        """Traverses the model and returns layers satisfying `stop_fn` criteria."""
        to_visit = set(source_layers)
        used_layers = set(source_layers)
        found_layers = set()
        while to_visit:
            layer = to_visit.pop()
            if stop_fn(layer):
                found_layers.add(layer)
            else:
                next_layers = next_fn(layer)
                if not next_layers:
                    return set()
                for next_layer in next_layers:
                    if next_layer not in used_layers:
                        used_layers.add(next_layer)
                        to_visit.add(next_layer)

        return found_layers


class OptimizerProcess:
    @staticmethod
    def prune_process(
            model_path: str,
            dm_json: bytes,
            batch_size: int,
            pruning_plan: bytes,
            pipe,
            problem_type: ModelProblemInt,
            output_model_path: Optional[str]
    ) -> tf.keras.Sequential:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        gpu = gpus[0]
        tf.config.experimental.set_memory_growth(gpu, True)
        model = tf.keras.models.load_model(model_path)
        dm = DatasetManager.fromJSON(dm_json)
        pruning_plan = PruningPlan.fromJSON(pruning_plan)
        epochs = 1
        train_ds, test_ds = dm.generate_batched_dataset(batch_size=batch_size)
        num_images = sum(map(lambda x: 1, train_ds))
        end_step = num_images * epochs
        pruning_schedule = pruning_plan.generate_pruning_schedule(end_step)
        pruning_params = {"pruning_schedule": pruning_schedule}
        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
            model, **pruning_params, pruning_policy=MyPolicy()
        )

        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
        ]
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-5  # model.optimizer.learning_rate.numpy()
        )
        try:
            from_logits = model.loss.get_config()["from_logits"]
        except AttributeError:
            from_logits = True
        if problem_type == ModelProblemInt.CATEGORICAL_CLASSIFICATION:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
            pruned_model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
        else:
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)
            pruned_model.compile(optimizer=optimizer, loss=loss, metrics=["binary_accuracy"])
        print(f"STARTING PRUNING WITH PR:{pruning_plan.targetSparsity}")
        pruned_model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=epochs,
            callbacks=callbacks,
            batch_size=batch_size,
        )
        if problem_type == ModelProblemInt.CATEGORICAL_CLASSIFICATION:
            val_accuracy = pruned_model.history.history["val_accuracy"]
        else:
            val_accuracy = pruned_model.history.history["val_binary_accuracy"]
        pipe.send(val_accuracy[-1])
        pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
        if output_model_path is not None:
            OptimizerProcess.static_save_model(pruned_model, output_model_path)
        return pruned_model

    @staticmethod
    def cluster_process(
            model_path,
            serial_dataset_manager: bytes,
            number_of_clusters: int,
            batch_size: int,
            pipe,
            problem_type: ModelProblemInt,
            output_model: Optional[str]
    ) -> None:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        gpu = gpus[0]
        tf.config.experimental.set_memory_growth(gpu, True)
        dataset_manager = DatasetManager.fromJSON(serial_dataset_manager)

        CentroidInitialization = tfmot.clustering.keras.CentroidInitialization
        clustering_params = {
            "preserve_sparsity": True
        }
        # Cluster a whole model
        cluster_weights = tfmot.clustering.keras.cluster_weights
        model: tf.keras.models.Sequential = tf.keras.models.load_model(model_path)
        try:
            clustered_model = cluster_weights(
                model,
                number_of_clusters=number_of_clusters,
                cluster_centroids_init=CentroidInitialization.KMEANS_PLUS_PLUS,
                **clustering_params,
            )
        except tf.errors.InvalidArgumentError as e:
            print("Cannot use KMEANS++, DENSITY BASED APPROACH IS USED")
            clustered_model = cluster_weights(
                model,
                number_of_clusters=number_of_clusters,
                cluster_centroids_init=CentroidInitialization.DENSITY_BASED,
                **clustering_params,
            )
        # Use smaller learning rate for fine-tuning clustered model
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-5  # model.optimizer.learning_rate.numpy()
        )
        if problem_type == ModelProblemInt.CATEGORICAL_CLASSIFICATION:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            clustered_model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
        else:
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            clustered_model.compile(optimizer=optimizer, loss=loss, metrics=["binary_accuracy"])
        # Fine-tune model
        train_ds, test_ds = dataset_manager.generate_batched_dataset(
            batch_size=batch_size
        )
        print(
            f"STARTING CLUSTERING FINE TUNING {number_of_clusters}"
        )
        clustered_model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=1,
            batch_size=batch_size,
        )
        if problem_type == ModelProblemInt.CATEGORICAL_CLASSIFICATION:
            val_accuracy = clustered_model.history.history["val_accuracy"]
        else:
            val_accuracy = clustered_model.history.history["val_binary_accuracy"]
        pipe.send(val_accuracy[-1])
        clustered_stripped_model = tfmot.clustering.keras.strip_clustering(
            clustered_model
        )
        if output_model is not None:
            OptimizerProcess.static_save_model(clustered_stripped_model, output_model)

    @staticmethod
    def qat_process(
            model_path,
            serial_dataset_manager: bytes,
            batch_size: int,
            problem_type: ModelProblemInt,
            output_model_path: str,
            original_acc: Optional[float]
    ) -> None:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        gpu = gpus[0]
        tf.config.experimental.set_memory_growth(gpu, True)
        dataset_manager = DatasetManager.fromJSON(serial_dataset_manager)
        model: tf.keras.Sequential = tf.keras.models.load_model(model_path)
        train_ds, test_ds = dataset_manager.generate_batched_dataset(
            batch_size=batch_size
        )

        quant_aware_annotate_model = tfmot.quantization.keras.quantize_annotate_model(model)
        pcqat_model = tfmot.quantization.keras.quantize_apply(
            quant_aware_annotate_model,
            tfmot.experimental.combine.Default8BitClusterPreserveQuantizeScheme(preserve_sparsity=True))

        if model.optimizer is None:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=1e-5  # model.optimizer.learning_rate.numpy()
            )
        else:
            optimizer = model.optimizer
        try:
            from_logits = model.loss.get_config()["from_logits"]
        except AttributeError:
            from_logits = True
        if problem_type == ModelProblemInt.CATEGORICAL_CLASSIFICATION:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
            pcqat_model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
        else:
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)
            pcqat_model.compile(optimizer=optimizer, loss=loss, metrics=["binary_accuracy"])
        # Fine-tune the model

        best_model_path = tempfile.mkdtemp()
        checkpoint = tf.keras.callbacks.ModelCheckpoint(best_model_path,
                                                        monitor="val_loss", mode="min",
                                                        save_best_only=True, verbose=1)
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            restore_best_weights=False,
        )

        callbacks = [early_stop, checkpoint]
        # if original_acc is not None:
        #     stop_threshold = MyThresholdCallback(original_acc)
        #     callbacks.append(stop_threshold)

        pcqat_model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=10,
            batch_size=batch_size,
            callbacks=callbacks
        )

        converter = tf.lite.TFLiteConverter.from_keras_model(pcqat_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(output_model_path, 'wb') as f:
            f.write(tflite_model)
            f.close()
        print(f"saving in {output_model_path}")

    @staticmethod
    def static_save_model(model: tf.keras.Sequential, path: str):
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        model.save(path)
