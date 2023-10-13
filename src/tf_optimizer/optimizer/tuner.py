import logging
import multiprocessing
import os
import pathlib
import sys
import tempfile
from datetime import datetime
from statistics import mean
from time import time

import tensorflow as tf
from tf_optimizer_core.benchmarker_core import BenchmarkerCore, Result

from tf_optimizer.benchmarker.benchmarker import Benchmarker
from tf_optimizer.benchmarker.utils import get_tflite_model_size
from tf_optimizer.configuration import Configuration
from tf_optimizer.dataset_manager import DatasetManager
from tf_optimizer.optimizer.optimization_param import (
    OptimizationParam,
    QuantizationLayerToPrune,
    QuantizationTechnique,
)
from tf_optimizer.optimizer.optimizer import Optimizer


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
            batchsize=32,
            optimized_model_path=None
    ) -> None:
        self.original_model = original_model
        self.dataset_manager = dataset
        self.batch_size = batchsize
        self.optimization_param = OptimizationParam()
        self.optimization_param.toggle_pruning(True)
        self.optimization_param.set_pruning_target_sparsity(0.5)
        self.optimization_param.toggle_quantization(True)
        self.optimization_param.set_number_of_cluster(16)
        self.optimization_param.toggle_clustering(True)
        self.optimized_model_path = optimized_model_path
        self.applied_prs = []
        self.no_cluster_prs = []
        self.max_cluster_fails = 0
        now = datetime.now()
        date_time = now.strftime("%m-%d-%Y-%H:%M:%S")
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            filename="logs/tuner{}.log".format(date_time),
            encoding="utf-8",
            level=logging.INFO,
        )
        logging.info(f"DS:{self.dataset_manager.get_path()}")
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        self.configuation = Configuration()
        layersToQuantize = self.configuation.getConfig("QUANTIZATION", "layers")
        if layersToQuantize == "ALL":
            self.optimization_param.set_quantized_layers(
                QuantizationLayerToPrune.AllLayers
            )
        else:
            self.optimization_param.set_quantized_layers(
                QuantizationLayerToPrune.OnlyDeepLayer
            )
        qTech = self.configuation.getConfig("QUANTIZATION", "type")
        if qTech == "QAT":
            print("Quantization Aware Training Selected")
            self.optimization_param.set_quantization_technique(
                QuantizationTechnique.QuantizationAwareTraining
            )
        elif qTech == "PTQ":
            print("Post Training Quantization Selected")
            self.optimization_param.set_quantization_technique(
                QuantizationTechnique.PostTrainingQuantization
            )
        else:
            print(f"QUANTIZATION TYPE:{qTech} NOT VALID")
            exit()

    @staticmethod
    def measure_keras_accuracy_process(
            model_path: str,
            dataset_manager: bytes,
            batch_size: int,
            lr: float,
            from_logits: bool,
            q: multiprocessing.Queue,
    ):
        print("Measuring keras model accuracy")
        print(f"model path:{model_path}")
        model = tf.keras.models.load_model(model_path)
        print(f"MODEL LOADED {model}")

        dm = DatasetManager.fromJSON(dataset_manager)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
        model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
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
                self.dataset_manager.get_validation_folder(), self.dataset_manager.scale
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
                ),
            )
            p.start()
            res = q.get()
            p.join()
            return res

    async def getOptimizedModel(
            self, model_path, targetAccuracy: float, percentagePrecision: float = 2.0
    ) -> tuple[bytes, Result]:
        iterations = 0
        pruningRatio = 0.5
        minPruningRatio = 0
        maxPruningRatio = 1
        tflite_model = None
        model_result = None
        reachedAccuracy = 0
        self.optimization_param.set_pruning_target_sparsity(pruningRatio)
        counter_back_direction = 0
        counter_forward_direction = 0
        if self.optimization_param.get_number_of_cluster() <= self.max_cluster_fails:
            self.optimization_param.toggle_clustering(False)
            logging.info(
                f"SINCE REQUIRED NUMBER OF CLUSTER {self.optimization_param.get_number_of_cluster()} IS BELOW {self.max_cluster_fails}, CLUSTERIZATION IS DISABLED"
            )
        while abs(reachedAccuracy - targetAccuracy) > percentagePrecision / 100:
            # Computing pruning rate
            if (
                    iterations == 0
                    and self.optimization_param.isClusteringEnabled()
                    and len(self.applied_prs) > 0
            ):
                pruningRatio = mean(self.applied_prs)
            elif (
                    iterations == 0
                    and not self.optimization_param.isClusteringEnabled()
                    and len(self.no_cluster_prs) > 0
            ):
                pruningRatio = mean(self.no_cluster_prs)
            elif iterations == 0 or (
                    iterations == 1
                    and (len(self.applied_prs) > 0 or len(self.no_cluster_prs) > 0)
            ):
                # If first iteration and applied_pr is empty
                # if mean of applied_pr has failed, so it starts from the middle
                pruningRatio = 0.5
            elif reachedAccuracy < targetAccuracy or tflite_model is None:  # Go left
                # Pruning ratio decreases
                maxPruningRatio = pruningRatio
                pruningRatio = (minPruningRatio + pruningRatio) / 2
                counter_back_direction += 1
                counter_forward_direction = 0
            else:  # Go rigth
                # Pruning ration increases
                minPruningRatio = pruningRatio
                pruningRatio = (maxPruningRatio + pruningRatio) / 2
                counter_forward_direction += 1
                counter_back_direction = 0

            # Apply optimizations
            self.optimization_param.set_pruning_target_sparsity(pruningRatio)
            tf.keras.backend.clear_session()
            optimizer = Optimizer(
                model_path,
                optimization_param=self.optimization_param,
                batch_size=self.batch_size,
                dataset_manager=self.dataset_manager,
                early_breakup_accuracy=targetAccuracy,
                logger=logging,
            )
            logging.info(f"Optimizing with PR of {pruningRatio}")
            tflite_model = optimizer.optimize()

            if tflite_model is None:
                logging.info("Early stopped")
            else:  # Accuracy is ok in pruning or/and clustering
                model_result = await self.test_model(tflite_model)
                reachedAccuracy = model_result.accuracy
                logging.info(f"Measured accuracy {reachedAccuracy}")

            if counter_back_direction > self.configuation.getConfig(
                    "CLUSTERING", "number_of_backstep_to_exit"
            ):
                # Accuracy is too high, clusterization disabled
                logging.info("Accuracy is too high, clusterization disabled")
                self.max_cluster_fails = self.optimization_param.get_number_of_cluster()
                self.optimization_param.toggle_clustering(False)
                return await self.getOptimizedModel(
                    model_path, targetAccuracy, percentagePrecision
                )
            iterations += 1
        logging.info(
            f"Found model with acc:{reachedAccuracy} in {iterations} iterations"
        )
        if self.optimization_param.isClusteringEnabled():
            # If accuracy is ok, add pr in the list
            self.applied_prs.append(pruningRatio)
        else:
            self.no_cluster_prs.append(pruningRatio)
        return tflite_model, model_result

    async def tune(self) -> bytes:
        # Get parameters from config
        right = self.configuation.getConfig("CLUSTERING", "max_clusters_number")
        left = self.configuation.getConfig("CLUSTERING", "min_clusters_numbers")
        delta_precision = self.configuation.getConfig("TUNER", "DELTA_PERCENTAGE")
        isTimePrioritized = (
                self.configuation.getConfig("TUNER", "second_priority") == "SPEED"
        )
        isClusteringEnabled = self.configuation.getConfig("CLUSTERING", "enabled")

        # Step 0, save original model
        original_model_path = tempfile.mkdtemp()
        self.original_model.save(original_model_path)

        model_performance = await self.test_model(original_model_path)
        targetAccuracy = model_performance.accuracy

        self.optimization_param.toggle_clustering(True)
        cached_result = {}

        while isClusteringEnabled and abs(left - right) > 2:
            left_third = int(left + (right - left) / 3)
            right_third = int(right - (right - left) / 3)
            tf.keras.backend.clear_session()
            logging.info(f"Optimizing with {left_third} clusters")

            if left_third in cached_result.keys():
                result_left = cached_result[left_third]
            else:
                self.optimization_param.set_number_of_cluster(left_third)
                self.optimization_param.toggle_clustering(True)
                _, result = await self.getOptimizedModel(
                    original_model_path,
                    targetAccuracy,
                    percentagePrecision=delta_precision,
                )
                result_left = result.time if isTimePrioritized else result.size
                cached_result[left_third] = result_left

            tf.keras.backend.clear_session()
            logging.info(f"Optimizing with {right_third} clusters")

            if right_third in cached_result.keys():
                result_right = cached_result[right_third]
            else:
                self.optimization_param.set_number_of_cluster(right_third)
                self.optimization_param.toggle_clustering(True)
                _, result = await self.getOptimizedModel(
                    original_model_path,
                    targetAccuracy,
                    percentagePrecision=delta_precision,
                )
                result_right = result.time if isTimePrioritized else result.size
                cached_result[right_third] = result_right

            logging.info(
                f"RESULT C:{result_left}|{left_third} - RESULT D:{result_right}|{right_third}"
            )
            if result_left > result_right:
                left = left_third
            else:
                right = right_third

            logging.info(f"NEXT EVALUATED LEFT:{left} - RIGHT:{right}")

        choosen_clusters = (right + left) / 2  # Should be the minimum

        self.optimization_param.set_number_of_cluster(int(choosen_clusters))
        self.optimization_param.toggle_clustering(True and isClusteringEnabled)
        optimized_model, _ = await self.getOptimizedModel(
            original_model_path, targetAccuracy, percentagePrecision=delta_precision
        )

        return optimized_model
