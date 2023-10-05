from sqlalchemy import create_engine, desc, asc
from sqlalchemy.orm import sessionmaker
from tf_optimizer.task_manager.task import Task, TaskStatus
from tf_optimizer import Base
from tf_optimizer.optimizer.tuner import Tuner
from tf_optimizer.dataset_manager import DatasetManager
from threading import Thread, Semaphore
from zipfile import ZipFile
from typing import Union
import tempfile
import shutil
import os
import requests
import tensorflow as tf
import multiprocessing
import asyncio


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
        self.db.query(Task).where(Task.status != TaskStatus.COMPLETED).update({"status": TaskStatus.PENDING})
        self.db.commit()
        multiprocessing.set_start_method('spawn')
        self.sem = Semaphore()
        self.check_task_to_process()

    def delete_table(self) -> None:
        Task.__table__.drop(self.engine)

    def add_task(self, t: Task) -> None:
        self.db.add(t)
        self.db.commit()
        self.check_task_to_process()

    def delete_task(self, id: int) -> int:
        removed_rows = self.db.query(Task).where(Task.id == id).delete()
        self.db.commit()
        return removed_rows

    def get_task_by_id(self, id: int) -> Union[Task, None]:
        return self.db.query(Task).where(Task.id == id).first()

    def get_last_task(self) -> Task:
        return self.db.query(Task).order_by(desc(Task.id)).first()

    def get_all_task(self) -> list[Task]:
        return self.db.query(Task).order_by(desc(Task.id)).all()

    def update_task_state(self, id_task: int, status: TaskStatus) -> int:
        updated_rows = self.db.query(Task).filter(Task.id == id_task).update({"status": status})
        self.db.commit()
        self.db.flush()
        self.check_task_to_process()
        return updated_rows

    """
    Process the oldest task, if there ins't a task it exits immediatelly
    """

    def check_task_to_process(self):
        if not self.sem.acquire(blocking=False):
            print("SEM ACUQIRED")
            return
        print("CHECKING TASK TO PROCESS")

        print(list(map(lambda x: x.status, self.get_all_task())))
        processing_task = self.db.query(Task).where(Task.status == TaskStatus.PROCESSING).first()

        if processing_task is not None:
            return

        print("NO PROCESSING TASKS")

        older_task = (
            self.db.query(Task)
            .where(Task.status == TaskStatus.PENDING)
            .order_by(asc(Task.id)).first()
        )

        if older_task is not None:
            print("PENDING TASK IS PRESENT")
            self.update_task_state(older_task.id, TaskStatus.PROCESSING)
            t = Thread(target=TaskManager.create_processing_process,
                       args=(older_task.to_json(), self.update_task_state, self.sem))
            t.start()

    @staticmethod
    def create_processing_process(t: bytes, tm, sem: Semaphore):
        p = multiprocessing.Process(
            target=TaskManager.process_task, args=(t,)
        )
        p.start()
        p.join()
        sem.release()
        if p.exitcode == 0:
            tm(1, TaskStatus.COMPLETED)
        else:
            tm(1, TaskStatus.FAILED)
        print("SEM RELEASED")

    # On dedicated process
    @staticmethod
    def process_task(data: bytes) -> None:
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
            print(
                "Cannot detect model input size, provides it manually using the parameter --image_size"
            )
            exit()

        img_shape = (detected_input_size[1], detected_input_size[2])
        dm = DatasetManager(dataset_folder, img_size=img_shape, scale=t.dataset_scale)
        tuner = Tuner(
            original_model,
            dm,
            batchsize=t.batch_size,
            optimized_model_path=t.generate_filename()
        )
        asyncio.run(tuner.tune())

        # Here its end
        shutil.rmtree(temp_workspace)
