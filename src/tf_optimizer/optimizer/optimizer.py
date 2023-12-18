import os
import pathlib
import shutil
import tempfile
from enum import IntEnum
from multiprocessing import Process, Pipe

import tensorflow as tf

from tf_optimizer.configuration import Configuration
from tf_optimizer.dataset_manager import DatasetManager
from tf_optimizer.optimizer.optimization_param import (
    OptimizationParam,
    QuantizationTechnique, ModelProblemInt,
)
from tf_optimizer.optimizer.optimizer_process import OptimizerProcess


class Optimizer:
    dataset = None
    force_clustering_sparcing_preserve = False  # Set to true if input model is pruned

    def __init__(
            self,
            model,
            dataset_manager: DatasetManager,
            optimization_param: OptimizationParam,
            model_problem: ModelProblemInt,
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
        self.model_problem = model_problem
        self.configuration = Configuration()
        self.delta_precision = self.configuration.getConfig("TUNER", "DELTA_PERCENTAGE")

    def __representative_dataset_gen__(self):
        for image_batch, labels_batch in (
                self.dataset_manager.generate_batched_dataset()[0].shuffle(16).take(16)
        ):
            yield [image_batch]

    def __save_model__(self, model: tf.keras.Sequential):
        OptimizerProcess.static_save_model(model, self.saved_model_path)

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
                target=OptimizerProcess.prune_process,
                args=(
                    self.saved_model_path,
                    self.dataset_manager.toJSON(),
                    self.batch_size,
                    self.optimizationParam.toJSON(),
                    sender,
                    self.model_problem
                ),
            )
            p.start()
            recevied_accuracy = receiver.recv()
            if (
                    self.early_breakup_accuracy is not None
                    and recevied_accuracy + (self.delta_precision/100) < self.early_breakup_accuracy
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
                target=OptimizerProcess.cluster_process,
                args=(
                    self.saved_model_path,
                    self.dataset_manager.toJSON(),
                    self.optimizationParam.toJSON(),
                    self.batch_size,
                    sender,
                    self.model_problem
                ),
            )
            p.start()
            recevied_accuracy = receiver.recv()
            if (
                    self.early_breakup_accuracy is not None
                    and recevied_accuracy + (self.delta_precision/100) < self.early_breakup_accuracy
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
                    target=OptimizerProcess.qat_process,
                    args=(
                        self.saved_model_path,
                        self.dataset_manager.toJSON(),
                        self.optimizationParam.toJSON(),
                        self.batch_size,
                        self.model_problem
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

    def __del__(self) -> None:
        shutil.rmtree(self.saved_model_path)
