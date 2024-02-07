import asyncio
import copy
import os.path
import sys
from enum import Enum, auto
from typing import List, Tuple

import tensorflow as tf
from prettytable import PrettyTable
from termcolor import colored
from tf_optimizer_core.benchmarker_core import BenchmarkerCore

from tf_optimizer.benchmarker.model_info import ModelInfo
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
        if edge_devices is None or len(edge_devices) == 0:
            ed = EdgeDevice("localhost", 0)
            ed.alias = "local"
            ed.id = 0
            self.edge_devices = [ed]
        else:
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
                    result = await self.core.test_model(file_path, model.name, progressBar)
                    result.node_id = edge_device.id
                else:
                    result = await edge_device.send_model(file_path, model.name)

                new_model = copy.deepcopy(model)
                new_model.time = result.time
                new_model.accuracy = result.accuracy
                results[str(result.node_id)].append(new_model)

            """
                asyncio_task = asyncio.create_task(task)
                created_task.append((asyncio_task, edge_device.id))

            for task in created_task:
                result = await task[0]
                device_id = task[1]
                # Append time
                model.time = result.time
                model.accuracy = result.accuracy

                results[str(device_id)].append(model)
                print(f"{device_id} in {result.accuracy}")
            """
        return results

    async def set_dataset(self, dataset: DatasetManager):
        self.__dataset = dataset
        dataset_path = self.__dataset.get_validation_folder()
        created_tasks = []
        for edge_device in self.edge_devices:
            if edge_device.is_local_node() and self.core is None:
                self.core = BenchmarkerCore(
                    dataset_path,
                    interval=dataset.scale,
                    use_multicore=self.use_multicore,
                    data_format=dataset.data_format
                )
            else:
                print(
                    f"SENDING DS {dataset_path} at {edge_device.ip_address}:{edge_device.port}"
                )
                task = asyncio.create_task(edge_device.send_dataset(dataset))
                created_tasks.append(task)

        for task in created_tasks:
            await task

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
