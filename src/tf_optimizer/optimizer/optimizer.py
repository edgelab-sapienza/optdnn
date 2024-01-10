import tempfile
from multiprocessing import Process, Pipe
from typing import Optional

import tensorflow as tf

from tf_optimizer.configuration import Configuration
from tf_optimizer.dataset_manager import DatasetManager
from tf_optimizer.optimizer.optimization_param import (
    ModelProblemInt, PruningPlan, QuantizationParameter,
)
from tf_optimizer.optimizer.optimizer_process import OptimizerProcess


class Optimizer:
    dataset = None
    force_clustering_sparsing_preserve = False  # Set to true if input model is pruned

    def __init__(
            self,
            dataset_manager: DatasetManager,
            model_problem: ModelProblemInt,
            batch_size=32,
            logger=None,
    ) -> None:
        self.batch_size = batch_size
        self.dataset_manager = dataset_manager
        self.logger = logger
        self.model_problem = model_problem
        self.configuration = Configuration()
        self.delta_precision = self.configuration.getConfig("TUNER", "DELTA_PERCENTAGE")

    def __representative_dataset_gen__(self):
        for image_batch, labels_batch in (
                self.dataset_manager.generate_batched_dataset()[0].shuffle(16).take(16)
        ):
            yield [image_batch]

    def prune_model(self, input_model: str, output_model: Optional[str], pruning_plan: PruningPlan) -> float:
        tf.keras.backend.clear_session()
        print("Starting new process")
        receiver, sender = Pipe()
        p = Process(
            target=OptimizerProcess.prune_process,
            args=(
                input_model,
                self.dataset_manager.toJSON(),
                self.batch_size,
                pruning_plan.toJSON(),
                sender,
                self.model_problem,
                output_model
            ),
        )
        p.start()
        received_accuracy = receiver.recv()
        p.join()
        p.close()
        return received_accuracy

    async def quantize_model(self, input_model: str, quantization_parameter: QuantizationParameter) -> bytes:
        if quantization_parameter.quantizationTechnique is QuantizationParameter.quantizationTechnique.QuantizationAwareTraining:
            print("Starting QAT")
            qat_model_path = tempfile.mktemp("quantized_model.tflite")
            p = Process(
                target=OptimizerProcess.qat_process,
                args=(
                    input_model,
                    self.dataset_manager.toJSON(),
                    self.batch_size,
                    self.model_problem,
                    qat_model_path
                ),
            )
            p.start()
            p.join()
            p.close()
            f = open(qat_model_path, mode="rb")
            model_bytes = f.read()
            f.close()
            return model_bytes
        else:
            converter = tf.lite.TFLiteConverter.from_saved_model(input_model)

            print("GENERATING REPR DATASET")
            converter.representative_dataset = tf.lite.RepresentativeDataset(
                self.__representative_dataset_gen__
            )
            print("BUILT IN INT8")
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            if quantization_parameter.quantization_has_in_out_int():
                print("HAS INT IN/OUT")
                # If also in and out layers are integers
                in_out_type = quantization_parameter.get_in_out_type()
                converter.inference_input_type = in_out_type
                converter.inference_output_type = in_out_type

            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            return converter.convert()

    def clusterize(self, input_model: str, output_model: Optional[str], number_of_clusters: int) -> float:
        tf.keras.backend.clear_session()
        receiver, sender = Pipe()
        p = Process(
            target=OptimizerProcess.cluster_process,
            args=(
                input_model,
                self.dataset_manager.toJSON(),
                number_of_clusters,
                self.batch_size,
                sender,
                self.model_problem,
                output_model
            ),
        )
        p.start()
        received_accuracy = receiver.recv()
        p.join()
        p.close()
        return received_accuracy

    # end process
