import asyncio
import multiprocessing
import os
import shutil
import tempfile
from threading import Thread
from typing import Union
from zipfile import ZipFile

import psutil
import requests
import tensorflow as tf
from sqlalchemy import create_engine, desc, asc
from sqlalchemy.orm import sessionmaker
from tf_optimizer_core.benchmarker_core import BenchmarkerCore

from tf_optimizer import Base
from tf_optimizer.benchmarker.benchmarker import Benchmarker
from tf_optimizer.benchmarker.model_info import ModelInfo
from tf_optimizer.dataset_manager import DatasetManager
from tf_optimizer.optimizer.tuner import Tuner
from tf_optimizer.task_manager.benchmark_result import BenchmarkResult
from tf_optimizer.task_manager.edge_device import EdgeDevice
from tf_optimizer.task_manager.process_error_code import ProcessErrorCode
from tf_optimizer.task_manager.task import Task, TaskStatus


class TaskManager:
    run_tasks = False

    def __init__(self, run_tasks) -> None:
        SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"
        # SQLALCHEMY_DATABASE_URL = "postgresql://user:password@postgresserver/db"
        self.engine = create_engine(
            SQLALCHEMY_DATABASE_URL,
            connect_args={
                "check_same_thread": False,
            },
        )
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        Base.metadata.create_all(self.engine)
        self.db = SessionLocal()
        self.run_tasks = run_tasks
        if self.run_tasks:
            self.db.query(Task).where(Task.status != TaskStatus.COMPLETED).update(
                {"status": TaskStatus.PENDING, "error_msg": None}
            )
            self.db.commit()
            self.check_task_to_process()

    def __del__(self):
        if self.db:
            return self.engine

    def close(self):
        self.db.close()

    def delete_table(self) -> None:
        Task.__table__.drop(self.engine)
        EdgeDevice.__table__.drop(self.engine)

    """
    Add a task and return the task with the auto parameters assigned
    """

    # TODO if nodes are empty use the local node in the local container
    def add_task(
            self,
            t: Task,
            nodes: list[tuple[str, int]] = [("localhost", 0)],
            base_url: str = None,
    ) -> Task:
        self.db.add(t)
        self.db.commit()
        self.db.flush()

        if len(nodes) == 0:
            nodes = [("localhost", 0)]

        for node in nodes:
            if node[0] == "localhost" and node[1] == 0:
                # It is the local node
                edge_device = EdgeDevice(node[0], node[1])
                edge_device.alias = "local"
            else:
                edge_device = EdgeDevice(node[0], node[1])
            edge_device.task_id = t.id
            self.db.add(edge_device)
        self.db.commit()

        if base_url is not None:
            # t.id is filled after flush
            download_url = f"{base_url}{t.id}/download"
            self.update_task_field(t.id, "download_url", download_url)
        if self.run_tasks:
            self.check_task_to_process()
        return t

    def delete_task(self, task_id: int) -> int:
        task = self.get_task_by_id(task_id)
        if task is not None and task.devices is not None:
            device_ids = list(map(lambda x: x.id, task.devices))
        else:
            device_ids = []
        removed_rows = self.db.query(Task).where(Task.id == task_id).delete()
        self.db.query(EdgeDevice).where(EdgeDevice.task_id == task_id).delete()

        if len(device_ids) > 0:
            self.db.query(BenchmarkResult).filter(
                BenchmarkResult.edge_id.in_(device_ids)
            ).delete()
        self.db.commit()
        return removed_rows

    def get_task_by_id(self, id: int) -> Union[Task, None]:
        return (
            self.db.query(Task)
            .select_from(Task)
            .join(EdgeDevice, EdgeDevice.task_id == Task.id)
            .where(Task.id == id)
            .first()
        )

    def get_last_task(self) -> Task:
        return (
            self.db.query(Task)
            .select_from(Task)
            .join(EdgeDevice, EdgeDevice.task_id == Task.id)
            .order_by(desc(Task.id))
            .first()
        )

    def get_all_task(self) -> list[Task]:
        return (
            self.db.query(Task)
            .select_from(Task)
            .join(EdgeDevice, EdgeDevice.task_id == Task.id)
            .order_by(desc(Task.id))
            .all()
        )

    def update_task_state(self, id_task: int, status: TaskStatus) -> int:
        updated_rows = self.update_task_field(id_task, "status", status)
        if self.run_tasks:
            self.check_task_to_process()
        return updated_rows

    def remove_results(self, id_task: int):
        task = self.get_task_by_id(id_task)
        if task is None:
            return
        device_ids = list(map(lambda x: x.id, task.devices))
        res = (
            self.db.query(BenchmarkResult)
            .filter(BenchmarkResult.edge_id.in_(device_ids))
            .delete()
        )
        self.db.commit()
        return res

    def update_task_pid(self, id_task: int, pid: int):
        return self.update_task_field(id_task, "pid", pid)

    def update_task_field(self, id_task: int, field: str, value: any):
        updated_rows = (
            self.db.query(Task).filter(Task.id == id_task).update({field: value})
        )
        self.db.commit()
        self.db.flush()
        return updated_rows

    def report_error(self, id_task: int, error: str):
        return self.update_task_field(id_task, "error_msg", error)

    def terminate_task(self, id_task: int):
        task = self.get_task_by_id(id_task)
        if (
                task is not None
                and task.pid is not None
                and task.status == TaskStatus.PROCESSING
        ):
            p = psutil.Process(task.pid)
            p.terminate()
            self.update_task_state(id_task, TaskStatus.FAILED)

    def add_result(self, edge_id: int, model_info: ModelInfo):
        benchmark_result = BenchmarkResult()
        benchmark_result.time = model_info.time
        benchmark_result.accuracy = model_info.accuracy
        benchmark_result.size = model_info.size
        benchmark_result.edge_id = edge_id
        benchmark_result.name = model_info.name

        self.db.add(benchmark_result)
        self.db.commit()

    """
    Process the oldest task, if there ins't a task it exits immediatelly
    """

    def check_task_to_process(self):
        processing_task = (
            self.db.query(Task).where(Task.status == TaskStatus.PROCESSING).first()
        )

        if processing_task is not None:
            return

        older_task = (
            self.db.query(Task)
            .where(Task.status == TaskStatus.PENDING)
            .order_by(asc(Task.id))
            .first()
        )

        if older_task is not None:
            self.update_task_state(older_task.id, TaskStatus.PROCESSING)
            t = Thread(
                target=TaskManager.create_processing_process,
                args=(older_task.to_json(),),
            )
            t.start()

    @staticmethod
    def create_processing_process(t: bytes):
        task: Task = Task.from_json(t)
        tm = TaskManager(run_tasks=False)
        tm.update_task_state(task.id, TaskStatus.PROCESSING)
        tm.close()

        p = multiprocessing.Process(
            target=TaskManager.process_task,
            args=(
                t,
            ),
        )
        p.start()
        p.join()
        tm = TaskManager(run_tasks=False)
        if p.exitcode == 0:
            tm.update_task_state(task.id, TaskStatus.COMPLETED)
            if task.callback_url is not None:
                payload = {"download_url": task.download_url}
                # A get request to the API
                requests.get(task.callback_url, params=payload)
        else:
            if p.exitcode == ProcessErrorCode.InputShapeNotDetectable:
                msg = "Cannot detect model input size, provides it manually using the parameter image_size"
                tm.report_error(task.id, msg)
            elif p.exitcode == ProcessErrorCode.WrongQuantizationType:
                msg = "Quantization type not supported, check config.ini file"
                tm.report_error(task.id, msg)
            elif p.exitcode == ProcessErrorCode.ConnectionRefused:
                msg = "Connection refused by the node"
                tm.report_error(task.id, msg)
            tm.update_task_state(task.id, TaskStatus.FAILED)
        tm.check_task_to_process()
        tm.close()

    # On dedicated process
    @staticmethod
    def process_task(data: bytes) -> None:
        tm = TaskManager(run_tasks=False)
        t = Task.from_json(data)
        temp_workspace = tempfile.mkdtemp()
        # Download model
        response = requests.get(t.model_url)
        model_path = os.path.join(temp_workspace, "model.keras")
        open(model_path, "wb").write(response.content)

        # Download dataset
        response = requests.get(t.dataset_url)
        dataset_zip = os.path.join(temp_workspace, "dataset.zip")
        open(dataset_zip, "wb").write(response.content)
        # unzip dataset
        dataset_folder = os.path.join(temp_workspace, "dataset")
        os.makedirs(dataset_folder)
        with ZipFile(dataset_zip, "r") as zObject:
            zObject.extractall(path=dataset_folder)

        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)

        original_model = tf.keras.models.load_model(model_path)
        img_size = t.img_size
        detected_input_size = original_model.input_shape
        if img_size is not None and img_size[0] is not None and img_size[1] is not None:
            img_size[0] = int(img_size[0])
            img_size[1] = int(img_size[1])
            detected_input_size = (
                None,
                img_size[0],
                img_size[1],
                detected_input_size[3],
            )
        if detected_input_size[1] is None and detected_input_size[2] is None:
            exit(ProcessErrorCode.InputShapeNotDetectable)

        img_shape = (detected_input_size[1], detected_input_size[2])
        dm = DatasetManager(dataset_folder, img_size=img_shape, scale=t.dataset_scale)
        test = False
        if not test:
            tuner = Tuner(
                original_model,
                dm,
                model_problem=t.model_problem,
                batchsize=t.batch_size,
                optimized_model_path=t.generate_filename(),
            )
            result = asyncio.run(tuner.tune())
            optimized_model = result
        else:
            # Quick test
            optimized_model = tf.lite.TFLiteConverter.from_keras_model(original_model)
            optimized_model.optimizations = [tf.lite.Optimize.DEFAULT]
            optimized_model = optimized_model.convert()

        bc = Benchmarker(edge_devices=t.devices)
        asyncio.run(bc.set_dataset(dm))
        bc.add_model(original_model, "original")
        bc.add_tf_lite_model(optimized_model, "optimized")

        results = asyncio.run(bc.benchmark())
        for device in t.devices:
            for result in results[str(device.id)]:
                tm.add_result(device.id, result)
        # bc.summary()
        # Here ends
        bc.clearAllModels()
        tm.close()
        del dm
        shutil.rmtree(temp_workspace)
