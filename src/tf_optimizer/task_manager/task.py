from sqlalchemy import Column, Integer, DateTime, String, JSON
from tf_optimizer import Base
import datetime
import json


class Task(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True, index=True)
    status = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.datetime.now())
    model_url = Column(String, nullable=False)
    dataset_url = Column(String, nullable=False)
    dataset_scale = Column(JSON, nullable=False)
    img_size = Column(JSON, nullable=True, default=None)
    remote_nodes = Column(JSON, nullable=True, default=None)
    callback_url = Column(String, nullable=False)
    batch_size = Column(Integer, nullable=False, default=32)

    def __str__(self) -> str:
        return (
            f"ID: {self.id}, status: {self.status}, created_at: {self.created_at}, dataset_scale: {self.dataset_scale}, "
            + f"model_url: {self.model_url}, dataset_url: {self.dataset_url}, img_size: {self.img_size}, "
            + f"remote_nodes: {self.remote_nodes}, callback_url: {self.callback_url}, batch_size: {self.batch_size}"
        )

    def to_json(self) -> dict:
        d = {}
        d["id"] = self.id
        d["status"] = self.status
        d["created_at"] = str(self.created_at)
        d["model_url"] = self.model_url
        d["dataset_url"] = self.dataset_url
        d["dataset_scale"] = self.dataset_scale
        d["img_size"] = self.img_size
        d["remote_nodes"] = self.remote_nodes
        d["batch_size"] = self.batch_size
        d["callback_url"] = self.callback_url

        return json.dumps(d)
