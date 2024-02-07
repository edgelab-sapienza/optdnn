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
from typing import Optional, Union

import tensorflow as tf

from tf_optimizer.benchmarker.benchmarker import Benchmarker
from tf_optimizer.benchmarker.utils import get_tflite_model_size
from tf_optimizer.configuration import Configuration
from tf_optimizer.dataset_manager import DatasetManager
from tf_optimizer.optimizer.optimization_param import (
    OptimizationParam,
    QuantizationLayerToQuantize,
    QuantizationTechnique, ModelProblemInt, PruningPlan, QuantizationParameter, QuantizationType,
)
from tf_optimizer.optimizer.optimizer import Optimizer
from tf_optimizer.task_manager.process_error_code import ProcessErrorCode
from tf_optimizer_core.benchmarker_core import BenchmarkerCore, Result
from tf_optimizer.task_manager.task import Task

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
        self.force_uint8 = False
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
        self.start_time = now
        logging.info(f"START AT: {self.start_time}")
        logging.info(f"DS:{self.dataset_manager.get_path()}")
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(f"INITIAL PARAMETERS {original_model.count_params()}")
        flops, macs = Tuner.net_flops(original_model)
        logging.info(f"INITIAL FLOPS {flops}")
        logging.info(f"INITIAL MACS {macs}")

        self.configuration = Configuration()

    @staticmethod
    def net_flops(model, table=False) -> tuple[float, float]:
        # Code from https://github.com/ckyrkou/Keras_FLOP_Estimator/blob/master/python_code/net_flops.py
        if table == True:
            print('%25s | %16s | %16s | %16s | %16s | %6s | %6s' % (
                'Layer Name', 'Input Shape', 'Output Shape', 'Kernel Size', 'Filters', 'Strides', 'FLOPS'))
            print('-' * 170)
        t_flops = 0
        t_macc = 0
        for l in model.layers:
            o_shape, i_shape, strides, ks, filters = ['', '', ''], ['', '', ''], [1, 1], [0, 0], [0, 0]
            flops = 0
            macc = 0
            name = l.name
            factor = 1000000
            if 'InputLayer' in str(l):
                i_shape = l.input.get_shape()[1:4].as_list()
                o_shape = i_shape
            if 'Reshape' in str(l):
                i_shape = l.input.get_shape()[1:4].as_list()
                o_shape = l.output.get_shape()[1:4].as_list()
            if 'Add' in str(l) or 'Maximum' in str(l) or 'Concatenate' in str(l):
                i_shape = l.input[0].get_shape()[1:4].as_list() + [len(l.input)]
                o_shape = l.output.get_shape()[1:4].as_list()
                flops = (len(l.input) - 1) * i_shape[0] * i_shape[1] * i_shape[2]
            if 'Average' in str(l) and 'pool' not in str(l):
                i_shape = l.input[0].get_shape()[1:4].as_list() + [len(l.input)]
                o_shape = l.output.get_shape()[1:4].as_list()
                flops = len(l.input) * i_shape[0] * i_shape[1] * i_shape[2]
            if 'BatchNormalization' in str(l):
                i_shape = l.input.get_shape()[1:4].as_list()
                o_shape = l.output.get_shape()[1:4].as_list()
                bflops = 1
                for i in range(len(i_shape)):
                    bflops *= i_shape[i]
                flops /= factor
            if 'Activation' in str(l) or 'activation' in str(l):
                i_shape = l.input.get_shape()[1:4].as_list()
                o_shape = l.output.get_shape()[1:4].as_list()
                bflops = 1
                for i in range(len(i_shape)):
                    bflops *= i_shape[i]
                flops /= factor
            if 'pool' in str(l) and ('Global' not in str(l)):
                i_shape = l.input.get_shape()[1:4].as_list()
                strides = l.strides
                ks = l.pool_size
                flops = ((i_shape[0] / strides[0]) * (i_shape[1] / strides[1]) * (ks[0] * ks[1] * i_shape[2]))
            if 'Flatten' in str(l):
                i_shape = l.input.shape[1:4].as_list()
                flops = 1
                out_vec = 1
                for i in range(len(i_shape)):
                    flops *= i_shape[i]
                    out_vec *= i_shape[i]
                o_shape = flops
                flops = 0
            if 'Dense' in str(l):
                print(l.input)
                i_shape = l.input.shape[1:4].as_list()[0]
                if (i_shape == None):
                    i_shape = out_vec
                o_shape = l.output.shape[1:4].as_list()
                flops = 2 * (o_shape[0] * i_shape)
                macc = flops / 2
            if 'Padding' in str(l):
                flops = 0
            if 'Global' in str(l):
                i_shape = l.input.get_shape()[1:4].as_list()
                flops = ((i_shape[0]) * (i_shape[1]) * (i_shape[2]))
                o_shape = [l.output.get_shape()[1:4].as_list(), 1, 1]
                out_vec = o_shape
            if 'Conv2D ' in str(l) and 'DepthwiseConv2D' not in str(l) and 'SeparableConv2D' not in str(l):
                strides = l.strides
                ks = l.kernel_size
                filters = l.filters
                i_shape = l.input.get_shape()[1:4].as_list()
                o_shape = l.output.get_shape()[1:4].as_list()
                if (filters == None):
                    filters = i_shape[2]
                flops = 2 * ((filters * ks[0] * ks[1] * i_shape[2]) * (
                        (i_shape[0] / strides[0]) * (i_shape[1] / strides[1])))
                macc = flops / 2
            if 'Conv2D ' in str(l) and 'DepthwiseConv2D' in str(l) and 'SeparableConv2D' not in str(l):
                strides = l.strides
                ks = l.kernel_size
                filters = l.filters
                i_shape = l.input.get_shape()[1:4].as_list()
                o_shape = l.output.get_shape()[1:4].as_list()
                if (filters == None):
                    filters = i_shape[2]
                flops = 2 * (
                        (ks[0] * ks[1] * i_shape[2]) * ((i_shape[0] / strides[0]) * (i_shape[1] / strides[1])))
                macc = flops / 2
            t_macc += macc
            t_flops += flops
            if table:
                print('%25s | %16s | %16s | %16s | %16s | %6s | %5.4f' % (
                    name, str(i_shape), str(o_shape), str(ks), str(filters), str(strides), flops))
        t_flops = t_flops / factor
        # t_macc
        return t_flops * (10 ** 6), t_macc

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

    async def test_model(self, model: Union[bytes, str, pathlib.PosixPath]) -> Result:
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
            try:
                lr: float = self.original_model.optimizer.learning_rate.numpy()
            except:
                lr = 1e-5
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

            old_pruning_ratio = pruning_ratio
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
            if abs(pruning_ratio - old_pruning_ratio) < self.configuration.getConfig("PRUNING",
                                                                                     "min_pruning_ratio_update"):
                break
            if pruning_ratio < self.configuration.getConfig("PRUNING", "min_pruning_ratio"):
                logging.info(f"PRUNING RATE TOO LOW {pruning_ratio}, PRUNING DISABLED")
                return input_model_path, 0

        pruned_model_path = tempfile.mkdtemp("pruned_model")
        pruning_plan.targetSparsity = pruning_ratio + self.configuration.getConfig("PRUNING", "final_step")
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
        if target_accuracy < self.configuration.getConfig("GENERAL", "min_accuracy"):
            # Probably there is a problem with the dataset
            logging.info(f"Accuracy to low: {target_accuracy}")
            exit(ProcessErrorCode.LowAccuracy)
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
            target_accuracy + self.configuration.getConfig("PRUNING", "delta_finding"),
            percentage_precision=self.configuration.getConfig("PRUNING", "delta_percentage")
        )

        # Step 3, clusterize the model
        clustered_model_path = await self.find_clustered_model(
            pruned_model_path,
            optimizer,
            percentage_precision=self.configuration.getConfig("CLUSTERING", "delta_percentage")
        )
        if clustered_model_path is None:
            logging.info(f"CLUSTERING DISABLED")
            clustered_model_path = pruned_model_path
        else:
            shutil.rmtree(pruned_model_path)

        m = tf.keras.models.load_model(clustered_model_path)
        logging.info(f"FINAL PARAMETERS {m.count_params()}")
        flops, macs = Tuner.net_flops(m)
        logging.info(f"FINAL FLOPS {flops}")
        logging.info(f"FINAL MACS {macs}")
        del m

        # Step 4, quantize the model
        logging.info(f"QUANTIZING {clustered_model_path}")
        quantized_model = await self.quantize_model(clustered_model_path, optimizer)
        shutil.rmtree(clustered_model_path)
        shutil.rmtree(original_model_path)
        elapsed = datetime.now() - self.start_time
        logging.info(f"END AT: {datetime.now()}")
        logging.info(f"ELAPSED {elapsed}")
        return quantized_model

    async def quantize_model(self, input_model_path: str, optimizer: Optimizer) -> bytes:
        quantization_parameter = QuantizationParameter()
        layers_to_quantize = self.configuration.getConfig("QUANTIZATION", "layers")
        if layers_to_quantize == "ALL":
            quantization_parameter.layers_to_quantize = QuantizationLayerToQuantize.AllLayers
            quantization_parameter.set_in_out_type(tf.uint8)
        else:
            quantization_parameter.layers_to_quantize = QuantizationLayerToQuantize.OnlyDeepLayer

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

        result = await self.test_model(input_model_path)
        target_accuracy = result.accuracy

        best_quantized_model: Optional[tuple[float, bytes]] = None
        q_types: list[QuantizationType] = [QuantizationType.ForceInt8]
        if not self.force_uint8:
            q_types += [QuantizationType.Standard, QuantizationType.AllFP16]
        # Test this quantization types in order, the first which matches the accuracy is used
        for qType in q_types:
            quantization_parameter.quantizationType = qType
            quantized_model: bytes = await optimizer.quantize_model(input_model_path, quantization_parameter)
            result = await self.test_model(quantized_model)
            logging.info(f"Quantization {qType} get {result.accuracy} | size {result.size} | time: {result.time}")
            if abs(result.accuracy - target_accuracy) < self.configuration.getConfig("QUANTIZATION",
                                                                                     "delta_percentage") / 100:
                best_quantized_model = (result.accuracy, quantized_model)
                break
            elif best_quantized_model is None or result.accuracy > best_quantized_model[0]:
                best_quantized_model = (result.accuracy, quantized_model)

        return best_quantized_model[1]

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

        max_obtained: Optional[tuple[int, float]] = None

        while is_clustering_enabled and abs(left - right) > 1:
            left_third: int = int(left + (right - left) / 3)
            right_third: int = int(right - (right - left) / 3)
            tf.keras.backend.clear_session()

            logging.info(f"L: Optimizing with {left_third} clusters")
            result_left = optimizer.clusterize(input_model_path, None, left_third)
            # result_left = await compute_model(input_model_path, left_third)
            is_left_valid = result_left > target_accuracy - percentage_precision / 100

            if max_obtained is None:
                max_obtained = (left_third, result_left)

            if left_third > max_obtained[0]:  # New value has more cluster than the max
                if result_left >= max_obtained[1]:  # New value is better than the old
                    max_obtained = (left_third, result_left)
                else:  # New value with more clusters is worste
                    return None

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
            is_right_valid = result_right > target_accuracy - percentage_precision / 100

            logging.info(
                f"Clustering with {right_third} clusters returns acc: {result_right}, target: {target_accuracy}, {is_right_valid}")

            if right_third > max_obtained[0]:  # New value has more cluster than the max
                if result_right >= max_obtained[1]:  # New value is better than the old
                    max_obtained = (right_third, result_right)
                else:  # New value with more clusters is worste
                    return None

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
