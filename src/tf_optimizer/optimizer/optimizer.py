import tensorflow as tf
import tensorflow_model_optimization as tfmot
import pathlib
from tf_optimizer.dataset_manager import DatasetManager
from tf_optimizer.optimizer.optimization_param import (
    OptimizationParam,
    QuantizationTechnique,
)
import tempfile
import os
import shutil
from multiprocessing import Process, Pipe
from tensorflow_model_optimization.python.core.sparsity.keras import prunable_layer
from tensorflow_model_optimization.python.core.sparsity.keras import prune_registry
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_policy import (
    PruningPolicy,
)


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


class Optimizer:
    dataset = None
    force_clustering_sparcing_preserve = False  # Set to true if input model is pruned

    def __init__(
        self,
        model,
        dataset_manager: DatasetManager,
        optimization_param: OptimizationParam,
        batch_size=32,
        early_breakup_accuracy=None,
        logger=None,
    ) -> None:
        self.saved_model_path = (
            tempfile.mktemp()
        )  # File name is created, since it doesn't have an extention the path can be used as folder
        if isinstance(model, str) or isinstance(model, pathlib.PosixPath):
            if os.path.isdir(model):
                shutil.copytree(model, self.saved_model_path)
            else:
                m = tf.keras.models.load_model(model)
                self.__save_model__(m)
                del m
        elif isinstance(model, tf.keras.Sequential):
            print("Sequential model deletected")
            self.__save_model__(model)
        self.batch_size = batch_size
        self.dataset_manager = dataset_manager
        self.optimizationParam = optimization_param
        self.early_breakup_accuracy = early_breakup_accuracy
        self.logger = logger

    def __representative_dataset_gen__(self):
        for image_batch, labels_batch in (
            self.dataset_manager.generate_batched_dataset()[0].shuffle(16).take(16)
        ):
            yield [image_batch]

    def __save_model__(self, model: tf.keras.Sequential):
        Optimizer.static_save_model(model, self.saved_model_path)

    @staticmethod
    def static_save_model(model: tf.keras.Sequential, path: str):
        if os.path.exists(path):
            shutil.rmtree(path)
        model.save(path)

    @staticmethod
    def prune_process(
        model_path: str,
        dm_json: bytes,
        batch_size: int,
        optimization_param: bytes,
        pipe,
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
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
        pruned_model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
        print("PUNING STARTED")
        pruned_model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=epochs,
            callbacks=callbacks,
            batch_size=batch_size,
        )
        val_accuracy = pruned_model.history.history["val_accuracy"]
        pipe.send(val_accuracy[-1])
        pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
        Optimizer.static_save_model(pruned_model, model_path)
        return pruned_model

    def optimize(self) -> bytes:
        """
        Apply optimizations and return a TFLite model
        """

        # Start process
        if self.optimizationParam.isPruningEnabled():
            tf.keras.backend.clear_session()
            print("Starting new process")
            receiver, sender = Pipe()
            p = Process(
                target=Optimizer.prune_process,
                args=(
                    self.saved_model_path,
                    self.dataset_manager.toJSON(),
                    self.batch_size,
                    self.optimizationParam.toJSON(),
                    sender,
                ),
            )
            p.start()
            recevied_accuracy = receiver.recv()
            if (
                self.early_breakup_accuracy is not None
                and recevied_accuracy < self.early_breakup_accuracy
            ):
                self.logger.info(f"PR: STOPPED WITH ACCURACY OF {recevied_accuracy}")
                p.kill()
                p.join()
                p.close()
                return None
            p.join()
            p.close()
        # end process

        # Start process
        if self.optimizationParam.isClusteringEnabled():
            tf.keras.backend.clear_session()
            receiver, sender = Pipe()
            p = Process(
                target=Optimizer.__cluster_process__,
                args=(
                    self.saved_model_path,
                    self.dataset_manager.toJSON(),
                    self.optimizationParam.toJSON(),
                    self.batch_size,
                    sender,
                ),
            )
            p.start()
            recevied_accuracy = receiver.recv()
            if (
                self.early_breakup_accuracy is not None
                and recevied_accuracy < self.early_breakup_accuracy
            ):
                self.logger.info(f"CL: STOPPED WITH ACCURACY OF {recevied_accuracy}")
                p.kill()
                p.join()
                p.close()
                return None
            p.join()
            p.close()
        # end process

        # Start process
        if self.optimizationParam.isQuantizationEnabled():
            isQAT = (
                self.optimizationParam.get_quantization_technique()
                is QuantizationTechnique.QuantizationAwareTraining
            )
            if isQAT:
                print("Starting QAT")
                p = Process(
                    target=Optimizer.__qat_process__,
                    args=(
                        self.saved_model_path,
                        self.dataset_manager.toJSON(),
                        self.optimizationParam.toJSON(),
                        self.batch_size,
                    ),
                )
                p.start()
                p.join()
                p.close()

            converter = tf.lite.TFLiteConverter.from_saved_model(self.saved_model_path)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            if not isQAT:
                print("GENERATING REPR DATASET")
                converter.representative_dataset = tf.lite.RepresentativeDataset(
                    self.__representative_dataset_gen__
                )
                print("BUILT IN INT8")
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                ]
                if self.optimizationParam.quantizationHasInOutInt():
                    print("HAS INT IN/OUT")
                    # If also in and out layers are integers
                    in_out_type = self.optimizationParam.get_in_out_type()
                    converter.inference_input_type = in_out_type
                    converter.inference_output_type = in_out_type
        else:
            converter = tf.lite.TFLiteConverter.from_saved_model(self.saved_model_path)
        return converter.convert()
        # end process

    @staticmethod
    def __cluster_process__(
        model_path,
        serial_dataset_manager: bytes,
        serial_optimization_param: bytes,
        batch_size: int,
        pipe,
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
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        clustered_model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
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
        val_accuracy = clustered_model.history.history["val_accuracy"]
        pipe.send(val_accuracy[-1])
        clustered_stripped_model = tfmot.clustering.keras.strip_clustering(
            clustered_model
        )
        Optimizer.static_save_model(clustered_stripped_model, model_path)

    @staticmethod
    def __qat_process__(
        model_path,
        serial_dataset_manager: bytes,
        serial_optimization_param: bytes,
        batch_size: int,
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
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
        model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
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
        Optimizer.static_save_model(model, model_path)

    def __del__(self) -> None:
        shutil.rmtree(self.saved_model_path)
