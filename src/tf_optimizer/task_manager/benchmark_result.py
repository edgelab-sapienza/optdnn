from sqlalchemy import Column, Integer, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, relationship

from tf_optimizer import Base


class BenchmarkResult(Base):
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, index=True)
    time = Column(Float)
    name = Column(String)
    accuracy = Column(Float)
    size = Column(Integer)
    edge_id = Column(ForeignKey("edge_result.id"))
    edge: Mapped["EdgeDevice"] = relationship(back_populates="results")
