from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker
from tf_optimizer.task_manager.task import Task
from tf_optimizer import Base


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

    def delete_table(self) -> None:
        Task.__table__.drop(self.engine)

    def add_task(self, t: Task) -> None:
        self.db.add(t)
        self.db.commit()

    def delete_task(self, id: int) -> int:
        removed_rows = self.db.query(Task).where(Task.id == id).delete()
        self.db.commit()
        return removed_rows

    def get_task_by_id(self, id: int) -> Task | None:
        return self.db.query(Task).where(Task.id == id).first()

    def get_last_task(self) -> Task:
        return self.db.query(Task).order_by(desc(Task.id)).first()

    def get_all_task(self) -> list[Task]:
        return self.db.query(Task).order_by(desc(Task.id)).all()
