import logging
import multiprocessing
import os
import pathlib
import shutil
import sys
import tempfile
from datetime import datetime
from statistics import mean
from time import time
from typing import Optional

import tensorflow as tf

from tf_optimizer.benchmarker.benchmarker import Benchmarker
from tf_optimizer.benchmarker.utils import get_tflite_model_size
from tf_optimizer.configuration import Configuration
from tf_optimizer.dataset_manager import DatasetManager
from tf_optimizer.optimizer.optimization_param import (
    OptimizationParam,
    QuantizationLayerToPrune,
    QuantizationTechnique, ModelProblemInt, PruningPlan, QuantizationParameter,
)
from tf_optimizer.optimizer.optimizer import Optimizer
from tf_optimizer.task_manager.process_error_code import ProcessErrorCode
from tf_optimizer_core.benchmarker_core import BenchmarkerCore, Result


class SpeedMeausureCallback(tf.keras.callbacks.Callback):
    current_batch_times = []
    start_time = 0

    def on_test_batch_begin(self, batch, logs=None):
        self.start_time = time()

    def on_test_batch_end(self, batch, logs=None):
        delta = time() - self.start_time
        delta = delta / 1000
        self.current_batch_times.append(delta)

    def get_avg_time(self):
        return mean(self.current_batch_times)


class Tuner:
    """
    This class is responsible to find the optimal parameters for the optimization
    """

    def __init__(
            self,
            original_model: tf.keras.Sequential,
            dataset: DatasetManager,
            model_problem: ModelProblemInt,
            batch_size=32,
    ) -> None:
        self.original_model = original_model
        self.dataset_manager = dataset
        self.batch_size = batch_size
        self.optimization_param = OptimizationParam()
        self.optimization_param.toggle_pruning(True)
        self.optimization_param.set_pruning_target_sparsity(0.5)
        self.optimization_param.toggle_quantization(True)
        self.optimization_param.set_number_of_cluster(16)
        self.optimization_param.toggle_clustering(True)
        self.applied_prs = []
        self.no_cluster_prs = []
        self.max_cluster_fails = 0
        self.model_problem = model_problem
        now = datetime.now()
        date_time = now.strftime("%m-%d-%Y-%H:%M:%S")
        LOGS_DIR = "logs"
        os.makedirs(LOGS_DIR, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(LOGS_DIR, f"tuner{date_time}.log"),
            encoding="utf-8",
            level=logging.INFO,
            force=True
        )
        logging.info(f"DS:{self.dataset_manager.get_path()}")
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        self.configuration = Configuration()

    @staticmethod
    def measure_keras_accuracy_process(
            model_path: str,
            dataset_manager: bytes,
            batch_size: int,
            lr: float,
            from_logits: bool,
            q: multiprocessing.Queue,
            model_problem: ModelProblemInt
    ):
        print("Measuring keras model accuracy")
        print(f"model path:{model_path}")
        model = tf.keras.models.load_model(model_path)
        print(f"MODEL LOADED {model}")

        dm = DatasetManager.fromJSON(dataset_manager)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        if model_problem == ModelProblemInt.CATEGORICAL_CLASSIFICATION:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
            model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
        else:
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)
            model.compile(optimizer=optimizer, loss=loss, metrics=["binary_accuracy"])

        speedCallback = SpeedMeausureCallback()
        metrics = model.evaluate(
            dm.generate_batched_dataset(batch_size)[1], callbacks=[speedCallback]
        )
        res = Result()
        res.time = speedCallback.get_avg_time()
        res.accuracy = metrics[1]
        q.put(res)

    async def test_model(self, model) -> Result:
        if isinstance(model, bytes):
            print("Measuring tflite model accuracy")
            bc = BenchmarkerCore(
                self.dataset_manager.get_validation_folder(), self.dataset_manager.scale, use_multicore=True,
                data_format=self.dataset_manager.data_format
            )
            result = await bc.test_model(
                model, callback=Benchmarker.OfflineProgressBar()
            )
            result.size = get_tflite_model_size(model)
            return result

        if isinstance(model, str) or isinstance(model, pathlib.PosixPath):
            model = str(model)
            print(f"Measuring {model}")
            # Start a new process
            lr: float = self.original_model.optimizer.learning_rate.numpy()
            try:
                from_logits: bool = self.original_model.loss.get_config()["from_logits"]
            except AttributeError:
                from_logits: bool = False
            q = multiprocessing.Queue()

            serialized_dm = self.dataset_manager.toJSON()
            p = multiprocessing.Process(
                target=Tuner.measure_keras_accuracy_process,
                args=(
                    model,
                    serialized_dm,
                    self.batch_size,
                    lr,
                    from_logits,
                    q,
                    self.model_problem
                ),
            )
            p.start()
            res = q.get()
            p.join()
            return res

    async def find_pruned_model(
            self, input_model_path: str, optimizer: Optimizer, target_accuracy: float, percentage_precision: float = 2.0
    ) -> tuple[str, float]:
        pruning_ratio = 0.5
        min_pruning_ratio = 0
        max_pruning_ratio = 1
        self.optimization_param.set_pruning_target_sparsity(pruning_ratio)
        pruning_plan = PruningPlan()
        pruning_plan.schedule = PruningPlan.schedule.PolynomialDecay
        logging.info(f"PR: Target accuracy of {target_accuracy}")

        while True:
            # Computing pruning rate
            # If first iteration and applied_pr is empty
            # if mean of applied_pr has failed, so it starts from the middle

            pruning_plan.targetSparsity = pruning_ratio
            tf.keras.backend.clear_session()

            logging.info(f"Optimizing with PR of {pruning_ratio}")
            reached_accuracy = optimizer.prune_model(input_model_path, None, pruning_plan)
            logging.info(f"PR:{pruning_ratio} returns an accuracy of {reached_accuracy}")
            logging.info(
                f"PR: Comparing reached {reached_accuracy}, with target:{target_accuracy}, diff:{abs(reached_accuracy - target_accuracy)} and delta:{percentage_precision / 100}")

            if abs(reached_accuracy - target_accuracy) <= percentage_precision / 100:
                break
            elif reached_accuracy < target_accuracy:  # Go left
                # Pruning ratio decreases
                max_pruning_ratio = pruning_ratio
                pruning_ratio = (min_pruning_ratio + pruning_ratio) / 2
            else:  # Go right
                # Pruning ratio increases
                min_pruning_ratio = pruning_ratio
                pruning_ratio = (max_pruning_ratio + pruning_ratio) / 2

        pruned_model_path = tempfile.mkdtemp("pruned_model")
        pruning_plan.targetSparsity = pruning_ratio - 0.03
        reached_accuracy = optimizer.prune_model(input_model_path, pruned_model_path, pruning_plan)
        logging.info(f"PR:{pruning_plan.targetSparsity} final accuracy {reached_accuracy}")
        return pruned_model_path, pruning_plan.targetSparsity

    async def get_optimized_model(self) -> bytes:
        # Step 0, save original model
        original_model_path = tempfile.mkdtemp()
        self.original_model.save(original_model_path)

        # Step 1, evaluate original model
        model_performance = await self.test_model(original_model_path)
        target_accuracy = model_performance.accuracy
        logging.info(f"Target accuracy: {target_accuracy}")

        delta_precision = self.configuration.getConfig("TUNER", "DELTA_PERCENTAGE")

        optimizer = Optimizer(
            model_problem=self.model_problem,
            batch_size=self.batch_size,
            dataset_manager=self.dataset_manager,
            logger=logging,
        )

        # Step 2, prune the model
        pruned_model_path, final_pruning_rate = await self.find_pruned_model(
            original_model_path,
            optimizer,
            target_accuracy - 0.01,
            percentage_precision=delta_precision)

        # Step 3, clusterize the model
        clustered_model_path = await self.find_clustered_model(
            pruned_model_path,
            optimizer,
            percentage_precision=delta_precision
        )
        if clustered_model_path is None:
            clustered_model_path = pruned_model_path
        else:
            shutil.rmtree(pruned_model_path)

        # Step 4, quantize the model
        quantized_model = await self.quantize_model(clustered_model_path, optimizer)
        shutil.rmtree(clustered_model_path)
        return quantized_model

    async def quantize_model(self, input_model_path: str, optimizer: Optimizer) -> bytes:
        quantization_parameter = QuantizationParameter()
        layers_to_quantize = self.configuration.getConfig("QUANTIZATION", "layers")
        if layers_to_quantize == "ALL":
            quantization_parameter.layersToPrune = QuantizationLayerToPrune.AllLayers
            quantization_parameter.set_in_out_type(tf.uint8)
        else:
            quantization_parameter.layersToPrune = QuantizationLayerToPrune.OnlyDeepLayer

        quantization_technique = self.configuration.getConfig("QUANTIZATION", "type")
        if quantization_technique == "QAT":
            print("Quantization Aware Training Selected")
            quantization_parameter.set_quantization_technique(QuantizationTechnique.QuantizationAwareTraining)

        elif quantization_technique == "PTQ":
            print("Post Training Quantization Selected")
            quantization_parameter.set_quantization_technique(QuantizationTechnique.PostTrainingQuantization)
        else:
            print(f"QUANTIZATION TYPE:{quantization_technique} NOT VALID")
            exit(ProcessErrorCode.WrongQuantizationType)

        quantized_model = await optimizer.quantize_model(input_model_path, quantization_parameter)
        return quantized_model

    async def find_clustered_model(self, input_model_path: str, optimizer: Optimizer,
                                   percentage_precision: float = 2.0) -> Optional[str]:
        # Get parameters from config
        right: int = self.configuration.getConfig("CLUSTERING", "max_clusters_number")
        left: int = self.configuration.getConfig("CLUSTERING", "min_clusters_numbers")
        is_clustering_enabled: bool = self.configuration.getConfig("CLUSTERING", "enabled")
        model_performance = await self.test_model(input_model_path)
        target_accuracy = model_performance.accuracy

        async def compute_model(input_model: str, number_of_clusters: int) -> float:
            output_model = tempfile.mkdtemp("temp_clust_model")
            optimizer.clusterize(input_model, output_model, number_of_clusters)
            q_model = await self.quantize_model(output_model, optimizer)
            result = await self.test_model(q_model)
            shutil.rmtree(output_model)
            return result.accuracy

        while is_clustering_enabled and abs(left - right) > 1:
            left_third: int = int(left + (right - left) / 3)
            right_third: int = int(right - (right - left) / 3)
            tf.keras.backend.clear_session()

            logging.info(f"L: Optimizing with {left_third} clusters")
            result_left = optimizer.clusterize(input_model_path, None, left_third)
            # result_left = await compute_model(input_model_path, left_third)
            is_left_valid = abs(result_left - target_accuracy) <= percentage_precision / 100

            logging.info(
                f"Clustering with {left_third} clusters returns acc: {result_left}, target: {target_accuracy}, {is_left_valid}")
            if is_left_valid:
                right = left_third
                logging.info(f"Step on left of {left_third}")
                continue

            tf.keras.backend.clear_session()

            logging.info(f"R: Optimizing with {right_third} clusters")
            result_right = optimizer.clusterize(input_model_path, None, right_third)
            # result_right = await compute_model(input_model_path, right_third)
            is_right_valid = abs(result_right - target_accuracy) <= percentage_precision / 100

            logging.info(
                f"Clustering with {right_third} clusters returns acc: {result_right}, target: {target_accuracy}, {is_right_valid}")

            if is_left_valid is False and is_right_valid is False:
                left = right_third
                logging.info(f"Step on right of {right_third}")
            elif is_left_valid is False and is_right_valid is True:
                # Step in the middle
                left = left_third
                right = right_third
                logging.info(f"Step between {left_third} and {right_third}")

        output_model_path: str = tempfile.mkdtemp("clustered_model")
        logging.info(f"FINAL RESULT:{right} - {output_model_path}")
        final_accuracy = optimizer.clusterize(input_model_path, output_model_path, right)
        if abs(final_accuracy - target_accuracy) < percentage_precision / 100:
            return output_model_path
        else:  # It is not possible to reach the accuracy constraint, so clustering is disabled
            shutil.rmtree(output_model_path)
            return None
