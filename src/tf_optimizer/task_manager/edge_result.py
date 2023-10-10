from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import Mapped, relationship

from tf_optimizer import Base


class EdgeResult(Base):
    __tablename__ = "edge_result"

    id = Column(Integer, primary_key=True, index=True)
    ip_address = Column(String)
    port = Column(Integer)
    inference_time = Column(Float, default=0)
    task_id = Column(ForeignKey("tasks.id"))

    task: Mapped["Task"] = relationship(back_populates="devices")

    def __init__(self, ip_address: str, port: int):
        self.ip_address = ip_address
        self.port = port

    def __str__(self):
        return f"{self.id} IP: {self.ip_address} - {self.port} - TASK ID:{self.task_id}"