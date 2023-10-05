import os.path
from sqlalchemy import Column, Integer, DateTime, String, JSON
from tf_optimizer import Base
from enum import IntEnum
import datetime
import pickle


class TaskStatus(IntEnum):
    PENDING = 0
    PROCESSING = 1
    COMPLETED = 2
    FAILED = 3


class Task(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True, index=True)
    status = Column(Integer, default=TaskStatus.PENDING.value)
    created_at = Column(DateTime, default=datetime.datetime.now())
    model_url = Column(String, nullable=False)
    dataset_url = Column(String, nullable=False)
    dataset_scale = Column(JSON, nullable=False)
    img_size = Column(JSON, nullable=True, default=None)
    remote_nodes = Column(JSON, nullable=True, default=None)
    callback_url = Column(String, nullable=False)
    batch_size = Column(Integer, nullable=False, default=32)

    def generate_filename(self) -> str:
        filename = f"task_{self.id}.tflite"
        folder = "optimized_models"
        return os.path.join(folder, filename)

    def __str__(self) -> str:
        return (
                f"ID: {self.id}, status: {self.status}, created_at: {self.created_at}, dataset_scale: {self.dataset_scale}, "
                + f"model_url: {self.model_url}, dataset_url: {self.dataset_url}, img_size: {self.img_size}, "
                + f"remote_nodes: {self.remote_nodes}, callback_url: {self.callback_url}, batch_size: {self.batch_size}, optimized_model_path: {self.optimized_model_path}"
        )

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Task):
            return False
        else:
            return (
                    self.id == __value.id
                    and self.status == __value.status
                    # and self.created_at == __value.created_at
                    and self.model_url == __value.model_url
                    and self.dataset_url == __value.dataset_url
                    and self.dataset_scale == __value.dataset_scale
                    and self.img_size == __value.img_size
                    and self.remote_nodes == __value.remote_nodes
                    and self.callback_url == __value.callback_url
                    and self.batch_size == __value.batch_size
                    and self.optimized_model_path == __value.optimized_model_path
            )

    def to_json(self) -> bytes:
        d = {}
        d["id"] = self.id
        d["status"] = self.status
        d["created_at"] = self.created_at
        d["model_url"] = self.model_url
        d["dataset_url"] = self.dataset_url
        d["dataset_scale"] = self.dataset_scale
        d["img_size"] = self.img_size
        d["remote_nodes"] = self.remote_nodes
        d["batch_size"] = self.batch_size
        d["callback_url"] = self.callback_url
        return pickle.dumps(d)

    @staticmethod
    def from_json(data: bytes):
        data = pickle.loads(data)
        t = Task()
        t.id = data["id"]
        t.status = data["status"]
        t.created_at = data["created_at"]
        t.model_url = data["model_url"]
        t.dataset_url = data["dataset_url"]
        t.dataset_scale = data["dataset_scale"]
        t.img_size = data["img_size"]
        t.remote_nodes = data["remote_nodes"]
        t.batch_size = data["batch_size"]
        t.callback_url = data["callback_url"]
        return t
