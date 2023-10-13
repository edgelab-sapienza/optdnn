import asyncio
import os.path
import sys
from enum import Enum, auto
from typing import List, Tuple

import tensorflow as tf
from prettytable import PrettyTable
from termcolor import colored
from tf_optimizer_core.benchmarker_core import BenchmarkerCore

from tf_optimizer.benchmarker.model_info import ModelInfo
from tf_optimizer.benchmarker.result import Result
from tf_optimizer.dataset_manager import DatasetManager
from tf_optimizer.task_manager.edge_device import EdgeDevice
from . import utils as utils


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
    edge_devices = None

    class OfflineProgressBar(BenchmarkerCore.Callback):
        async def progress_callback(
                self, acc: float, progress: float, took_time: float, model_name: str = ""
        ):
            current_accuracy = "{0:.2f}".format(acc)
            formatted_took_time = "{0:.2f}".format(took_time)
            print(
                f"\rBenchmarking: {model_name} - progress: {int(progress)}% - accuracy: {current_accuracy}% - speed: {formatted_took_time} ms",
                end="",
            )
            sys.stdout.flush()

    def __init__(
            self,
            edge_devices: list[EdgeDevice],
            use_multicore=True,
    ) -> None:
        self.use_multicore = use_multicore
        self.edge_devices = edge_devices

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
        print(f"ADDING {name} model")
        """
        Add a model to the benchmark system
        :param bytes model: Tensorflow lite model
        :param str name: Model's name
        :param bool is_reference: True to set the model as reference for others models
        """
        _model = ModelInfo(model, name, is_reference)
        self.models.append(_model)

    async def benchmark(self) -> dict:
        """
        Benchmark inserted models
        :param model: Unbatch dataset composed by elements of (input, label)
        """
        results = {}
        if self.edge_devices is None and self.core is None:
            print("You must first call set_dataset_path")
            return

        for model in self.models:
            file_path = model.get_model_path()
            model.size = utils.get_gzipped_model_size(file_path)
            created_task: List[Tuple[asyncio.Task, int]] = []

            for edge_device in self.edge_devices:
                if str(edge_device.id) not in results.keys():
                    results[str(edge_device.id)] = []

                # Start local computing
                if edge_device.is_local_node():
                    progressBar = Benchmarker.OfflineProgressBar()
                    task = self.core.test_model(file_path, model.name, progressBar)
                else:
                    task = edge_device.send_model(file_path, model.name)

                asyncio_task = asyncio.create_task(task)
                created_task.append((asyncio_task, edge_device.id))

            for task in created_task:
                result = await task[0]
                device_id = task[1]
                # Append time
                model.time = result.time
                model.accuracy = result.accuracy

                results[str(device_id)].append(model)

        return results

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
            took_time = model.time
            isFastest = took_time == min(map(lambda x: x.time, self.models))
            isSlowest = took_time == max(map(lambda x: x.time, self.models))
            took_time_str = "{0:.4f}".format(took_time)
            if isFastest:
                took_time_str = colored(took_time_str, "green")
            elif isSlowest:
                took_time_str = colored(took_time_str, "red")
            elements.append(took_time_str)

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
        for edge_device in self.edge_devices:
            if edge_device.is_local_node() and self.core is None:
                self.core = BenchmarkerCore(
                    dataset_path,
                    interval=dataset.scale,
                    use_multicore=self.use_multicore,
                )
            else:
                print(
                    f"SENDING DS {dataset_path} at {edge_device.ip_address}:{edge_device.port}"
                )
                await edge_device.send_dataset(dataset_path)

    async def clear_online_node(self):
        if self.edge_devices is not None:
            for edge_device in self.edge_devices:
                await edge_device.close()

    def clearAllModels(self) -> None:
        for model in self.models:
            path = model.get_model_path()
            if os.path.exists(path):
                os.remove(path)
        self.models.clear()
