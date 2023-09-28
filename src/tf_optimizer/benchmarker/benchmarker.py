from tf_optimizer.network.client import Client
from tf_optimizer.benchmarker.model_info import ModelInfo
from tf_optimizer_core.benchmarker_core import BenchmarkerCore
from tf_optimizer.dataset_manager import DatasetManager
from . import utils as utils
from tf_optimizer.benchmarker.result import Result
from enum import Enum, auto
from typing import List
from termcolor import colored
from prettytable import PrettyTable
import sys
import os
import tensorflow as tf
import hashlib


class Ordering(Enum):
    Asc = auto()
    Desc = auto()


class FieldToOrder(Enum):
    Name = auto()
    Time = auto()
    Accuracy = auto()
    Size = auto()
    InsertedOrder = auto()


class Benchmarker:
    models = []
    __dataset: DatasetManager = None
    core = None
    client = None
    result_cache = {}

    class OfflineProgressBar(BenchmarkerCore.Callback):
        async def progress_callback(
            self, acc: float, progress: float, tooked_time: float, model_name: str = ""
        ):
            current_accuracy = "{0:.2f}".format(acc)
            formatted_tooked_time = "{0:.2f}".format(tooked_time)
            print(
                f"\rBenchmarking: {model_name} - progress: {int(progress)}% - accuracy: {current_accuracy}% - speed: {formatted_tooked_time} ms",
                end="",
            )
            sys.stdout.flush()

    def __init__(
        self, use_remote_nodes=False, client: Client = None, use_multicore=True
    ) -> None:
        self.isOnline = use_remote_nodes
        self.use_multicore = use_multicore
        self.client = client

    def add_model(
        self, model: tf.keras.Sequential, name: str, is_reference: bool = False
    ) -> None:
        """
        Add a model to the benchmark system
        :param tf.keras.Sequential model: Tensorflow Sequential model
        :param str name: Model's name
        :param bool is_reference: True to set the model as reference for others models
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converted_model = converter.convert()
        self.add_tf_lite_model(converted_model, name, is_reference)

    def add_tf_lite_model(
        self, model: bytes, name: str, is_reference: bool = False
    ) -> None:
        """
        Add a model to the benchmark system
        :param bytes model: Tensorflow lite model
        :param str name: Model's name
        :param bool is_reference: True to set the model as reference for others models
        """
        _model = ModelInfo(model, name, is_reference)
        self.models.append(_model)

    async def benchmark(self) -> None:
        """
        Benchmark inserted models
        :param model: Unbatch dataset composed by elements of (input, label)
        """

        if self.client is None and self.core is None:
            print("You must first call set_dataset_path")
            return

        for model in self.models:
            file_path = model.get_model_path()
            model.size = utils.get_gzipped_model_size(file_path)

            # Start local computing
            m = hashlib.sha256()
            m.update(model.get_model())
            hash_code = m.digest().hex()
            if hash_code in self.result_cache.keys():
                res = self.result_cache[hash_code]
            else:
                progressBar = Benchmarker.OfflineProgressBar()
                if self.isOnline:
                    res = await self.client.send_model(file_path, model.name)
                else:
                    res = await self.core.test_model(file_path, model.name, progressBar)
                self.result_cache[hash_code] = res

            model.time = res.time
            model.accuracy = res.accuracy

            os.remove(file_path)
            print()

    def summary(
        self,
        fieldToOrder: FieldToOrder = FieldToOrder.InsertedOrder,
        order: Ordering = Ordering.Asc,
    ) -> List[Result]:
        results = []

        has_reference = any(map(lambda x: x.is_reference, self.models))
        if has_reference:
            reference = list(filter(lambda x: x.is_reference, self.models))
            slowest_time = reference[0].time
        else:
            slowest_time = max(map(lambda x: x.time, self.models))

        x = PrettyTable()
        # Build table header
        x.field_names = ["ID", "Model", "time (ms)", "Speedup", "Accuracy", "Size"]

        # Order elements
        isDescendOrder = order == Ordering.Desc
        if fieldToOrder == FieldToOrder.Accuracy:
            self.models.sort(key=lambda e: e.accuracy, reverse=isDescendOrder)
        elif fieldToOrder == FieldToOrder.Name:
            self.models.sort(key=lambda e: e.name, reverse=isDescendOrder)
        elif fieldToOrder == FieldToOrder.Size:
            self.models.sort(key=lambda e: e.size, reverse=isDescendOrder)
        elif fieldToOrder == FieldToOrder.Time:
            self.models.sort(key=lambda e: e.time, reverse=isDescendOrder)

        index = 0
        for model in self.models:
            elements = []
            elements.append(str(index))
            elements.append(model.name)
            result = Result(model, id=index)
            index += 1

            # Append time
            tooked_time = model.time
            isFastest = tooked_time == min(map(lambda x: x.time, self.models))
            isSlowest = tooked_time == max(map(lambda x: x.time, self.models))
            tooked_time_str = "{0:.4f}".format(tooked_time)
            if isFastest:
                tooked_time_str = colored(tooked_time_str, "green")
            elif isSlowest:
                tooked_time_str = colored(tooked_time_str, "red")
            elements.append(tooked_time_str)

            # Append speedup
            speedup = slowest_time / model.time
            speedup_str = str("{0:.2f}".format(speedup)) + "x"
            if isFastest:
                speedup_str = colored(speedup_str, "green")
            elif isSlowest:
                speedup_str = colored(speedup_str, "red")
            elements.append(speedup_str)
            result.speedup = speedup

            # Append accuracy
            accuracy_str = "{0:.2f}%".format(model.accuracy * 100)
            if model.accuracy == max(map(lambda x: x.accuracy, self.models)):  # Better
                accuracy_str = colored(accuracy_str, "green")
            elif model.accuracy == min(map(lambda x: x.accuracy, self.models)):  # Worse
                accuracy_str = colored(accuracy_str, "red")
            elements.append(accuracy_str)

            size = model.size
            # Append model size
            size_str = utils.sizeof_fmt(size)
            if size == max(map(lambda x: x.size, self.models)):  # Larger
                size_str = colored(size_str, "red")
            elif size == min(map(lambda x: x.size, self.models)):  # Smaller
                size_str = colored(size_str, "green")
            elements.append(size_str)
            results.append(result)

            x.add_row(elements)

        print(x)
        # Returns a list with the results
        return results

    async def set_dataset(self, dataset: DatasetManager):
        self.__dataset = dataset
        dataset_path = self.__dataset.get_validation_folder()
        if self.isOnline:
            if self.client is None:
                raise Exception(
                    "You want to use a client, but it is None in the costructor"
                )
            await self.client.send_dataset(dataset_path)
        else:
            self.core = BenchmarkerCore(
                dataset_path, interval=dataset.scale, use_multicore=self.use_multicore
            )

    async def clear_online_node(self):
        if self.isOnline and self.client is not None:
            await self.client.close()

    def clearAllModels(self) -> None:
        self.models.clear()
