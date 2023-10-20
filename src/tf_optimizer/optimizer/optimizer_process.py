import os
import shutil

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.sparsity.keras import prunable_layer, prune_registry
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_policy import PruningPolicy

from tf_optimizer.dataset_manager import DatasetManager
from tf_optimizer.optimizer.optimization_param import OptimizationParam, ModelProblemInt

"""
This class contains all the static code that will be spawned on dedicated processes for the optimization
"""


class MyPolicy(PruningPolicy):
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
            optimization_param: bytes,
            pipe,
            problem_type: ModelProblemInt
    ) -> tf.keras.Sequential:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        gpu = gpus[0]
        tf.config.experimental.set_memory_growth(gpu, True)
        model = tf.keras.models.load_model(model_path)
        dm = DatasetManager.fromJSON(dm_json)
        op = OptimizationParam.fromJSON(optimization_param)
        epochs = 1
        train_ds, test_ds = dm.generate_batched_dataset(batch_size=batch_size)
        num_images = sum(map(lambda x: 1, train_ds))
        end_step = num_images * epochs
        pruning_schedule = op.generate_pruning_schedule(end_step)
        pruning_params = {"pruning_schedule": pruning_schedule}
        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
            model, **pruning_params, pruning_policy=MyPolicy()
        )

        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
        ]
        try:
            lr = model.optimizer.learning_rate.numpy()
        except AttributeError:
            lr = 1e-5
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
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
        print("PUNING STARTED")
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
        OptimizerProcess.static_save_model(pruned_model, model_path)
        return pruned_model

    @staticmethod
    def cluster_process(
            model_path,
            serial_dataset_manager: bytes,
            serial_optimization_param: bytes,
            batch_size: int,
            pipe,
            problem_type: ModelProblemInt
    ) -> None:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        gpu = gpus[0]
        tf.config.experimental.set_memory_growth(gpu, True)
        model = tf.keras.models.load_model(model_path)
        optimizationParam = OptimizationParam.fromJSON(serial_optimization_param)
        dataset_manager = DatasetManager.fromJSON(serial_dataset_manager)
        force_clustering_sparcing_preserve = False

        CentroidInitialization = tfmot.clustering.keras.CentroidInitialization
        clustering_params = {
            "preserve_sparsity": optimizationParam.isPruningEnabled()
                                 or force_clustering_sparcing_preserve,
        }
        # Cluster a whole model
        cluster_weights = tfmot.clustering.keras.cluster_weights
        clustered_model = cluster_weights(
            model,
            number_of_clusters=optimizationParam.get_number_of_cluster(),
            cluster_centroids_init=CentroidInitialization.KMEANS_PLUS_PLUS,
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
            f"STARTING CLUSTERING FINE TUNING {optimizationParam.get_number_of_cluster()}"
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
        OptimizerProcess.static_save_model(clustered_stripped_model, model_path)

    @staticmethod
    def qat_process(
            model_path,
            serial_dataset_manager: bytes,
            serial_optimization_param: bytes,
            batch_size: int,
            problem_type: ModelProblemInt
    ) -> None:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        gpu = gpus[0]
        tf.config.experimental.set_memory_growth(gpu, True)
        dataset_manager = DatasetManager.fromJSON(serial_dataset_manager)
        optimizationParam = OptimizationParam.fromJSON(serial_optimization_param)
        # q_aware stands for for quantization aware.
        model = tf.keras.models.load_model(model_path)

        def apply_quantization_to_layer(layer):
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                print(f"SKIPPED QAT {layer}")
                return layer
            else:
                return tfmot.quantization.keras.quantize_annotate_layer(layer)

        model = tf.keras.models.clone_model(
            model,
            clone_function=apply_quantization_to_layer,
        )

        model = tfmot.quantization.keras.quantize_apply(
            model,
            tfmot.experimental.combine.Default8BitClusterPreserveQuantizeScheme(
                preserve_sparsity=optimizationParam.isPruningEnabled()
            ),
        )

        try:
            lr = model.optimizer.learning_rate.numpy()
        except AttributeError:
            lr = 1e-5
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        try:
            from_logits = model.loss.get_config()["from_logits"]
        except AttributeError:
            from_logits = True
        if problem_type == ModelProblemInt.CATEGORICAL_CLASSIFICATION:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
            model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
        else:
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)
            model.compile(optimizer=optimizer, loss=loss, metrics=["binary_accuracy"])
        # Fine-tune the model
        train_ds, test_ds = dataset_manager.generate_batched_dataset(
            batch_size=batch_size
        )
        model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=1,
            batch_size=batch_size,
        )
        OptimizerProcess.static_save_model(model, model_path)

    @staticmethod
    def static_save_model(model: tf.keras.Sequential, path: str):
        if os.path.exists(path):
            shutil.rmtree(path)
        model.save(path)
