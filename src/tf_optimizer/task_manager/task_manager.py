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

from tf_optimizer import Base
from tf_optimizer.dataset_manager import DatasetManager
from tf_optimizer.optimizer.tuner import Tuner
from tf_optimizer.task_manager.edge_result import EdgeResult
from tf_optimizer.task_manager.task import Task, TaskStatus


class TaskManager:
    def __init__(self) -> None:
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
        self.db.query(Task).where(Task.status != TaskStatus.COMPLETED).update(
            {"status": TaskStatus.PENDING, "error_msg": None}
        )
        self.db.commit()
        self.check_task_to_process()

    def delete_table(self) -> None:
        Task.__table__.drop(self.engine)
        EdgeResult.__table__.drop(self.engine)

    """
    Add a task and return the task with the auto parameters assigned
    """

    def add_task(self, t: Task, base_url: str = None) -> Task:
        self.db.add(t)
        self.db.commit()
        self.db.flush()

        if base_url is not None:
            # t.id is filled after flush
            download_url = f"{base_url}{t.id}/download"
            self.update_task_field(t.id, "download_url", download_url)
        self.check_task_to_process()
        return t

    def add_result(self, task: Task, ip_address: str, port: int):
        self.db.refresh(task)
        edge_result = EdgeResult(ip_address, port)
        edge_result.task_id = task.id
        self.db.add(edge_result)
        self.db.commit()

    def delete_task(self, id: int) -> int:
        removed_rows = self.db.query(Task).where(Task.id == id).delete()
        self.db.commit()
        return removed_rows

    def get_task_by_id(self, id: int) -> Union[Task, None]:
        return self.db.query(Task).select_from(Task).join(EdgeResult, EdgeResult.task_id == Task.id).where(
            Task.id == id).first()

    def get_last_task(self) -> Task:
        return self.db.query(Task).select_from(Task).join(EdgeResult, EdgeResult.task_id == Task.id).order_by(
            desc(Task.id)).first()

    def get_all_task(self) -> list[Task]:
        return self.db.query(Task).select_from(Task).join(EdgeResult, EdgeResult.task_id == Task.id).order_by(
            desc(Task.id)).all()

    def update_task_state(self, id_task: int, status: TaskStatus) -> int:
        updated_rows = self.update_task_field(id_task, "status", status)
        self.check_task_to_process()
        return updated_rows

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
                args=(
                    older_task.to_json(),
                    self.update_task_state,
                    self.update_task_pid,
                    self.report_error,
                ),
            )
            t.start()

    @staticmethod
    def create_processing_process(
            t: bytes, assign_status_callback, assign_pid_callback, report_error_fn
    ):
        task: Task = Task.from_json(t)
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=TaskManager.process_task,
            args=(
                t,
                queue,
            ),
        )
        p.start()
        assign_pid_callback(task.id, p.pid)
        p.join()
        if p.exitcode == 0:
            assign_status_callback(task.id, TaskStatus.COMPLETED)
            if task.callback_url is not None:
                payload = {"download_url": task.download_url}
                # A get request to the API
                requests.get(task.callback_url, params=payload)
        else:
            error_msg = queue.get()
            if error_msg is not None and type(error_msg) is str:
                report_error_fn(task.id, error_msg)
            assign_status_callback(task.id, TaskStatus.FAILED)

    # On dedicated process
    @staticmethod
    def process_task(data: bytes, queue: multiprocessing.Queue) -> None:
        t = Task.from_json(data)
        temp_workspace = tempfile.mkdtemp()
        # Download model
        response = requests.get(t.model_url)
        model_path = os.path.join(temp_workspace, "model.keras")
        open(model_path, "wb").write(response.content)

        # Download model
        response = requests.get(t.dataset_url)
        dataset_zip = os.path.join(temp_workspace, "dataset.zip")
        open(dataset_zip, "wb").write(response.content)
        # unzip dataset
        dataset_folder = os.path.join(temp_workspace, "dataset")
        os.makedirs(dataset_folder)
        with ZipFile(dataset_zip, "r") as zObject:
            zObject.extractall(path=dataset_folder)

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
            msg = "Cannot detect model input size, provides it manually using the parameter image_size"
            queue.put(msg)
            exit(-1)

        img_shape = (detected_input_size[1], detected_input_size[2])
        dm = DatasetManager(dataset_folder, img_size=img_shape, scale=t.dataset_scale)
        tuner = Tuner(
            original_model,
            dm,
            batchsize=t.batch_size,
            optimized_model_path=t.generate_filename(),
        )
        asyncio.run(tuner.tune())

        # Here its end
        shutil.rmtree(temp_workspace)
