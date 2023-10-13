from tf_optimizer.task_manager.task import Task, TaskStatus
from tf_optimizer.task_manager.task_manager import TaskManager


class TestClass:
    def setup_method(self) -> None:
        self.tm = TaskManager()

    def __add_task__(self) -> Task:
        t = Task()
        t.model_url = "http://127.0.0.1:9000/fashion_mnist.keras"
        t.dataset_url = "http://127.0.0.1:9000/fashion.zip"
        t.dataset_scale = [-1, 1]
        t.callback_url = "http://127.0.0.1:9000"
        t.status = TaskStatus.FAILED
        self.tm.add_task(t)
        return t

    def test_insertion(self) -> None:
        inserted_task = self.__add_task__()
        last_task = self.tm.get_last_task()
        assert inserted_task == last_task

    def test_task_deletion(self) -> None:
        inserted_task0 = self.__add_task__()
        inserted_task1 = self.__add_task__()
        assert self.tm.delete_task(inserted_task0.id) == 1
        assert self.tm.get_task_by_id(inserted_task0.id) is None
        assert self.tm.get_task_by_id(inserted_task1.id) is not None
        self.tm.delete_task(inserted_task1.id)

    def test_all_task(self) -> None:
        added_task = []
        for e in range(5):
            t = self.__add_task__()
            added_task.append(t)
        task_list = self.tm.get_all_task()
        assert len(task_list) == 5
        first_task = self.tm.get_last_task()
        assert task_list[0] == first_task

    def test_status_update(self) -> None:
        inserted_task = self.__add_task__()
        self.tm.update_task_state(inserted_task.id, TaskStatus.COMPLETED)
        updated_task = self.tm.get_task_by_id(inserted_task.id)
        assert updated_task.status == TaskStatus.COMPLETED

    def test_serialization(self) -> None:
        t1 = Task()
        t1.model_url = "URL DI PROVA"
        t1.dataset_url = "URL DI PROVA DATASET"
        t1.dataset_scale = "[-1,1]"
        t1.callback_url = "URL CALLBACK"
        t1.img_size = [200, 200]
        self.tm.add_task(t1, [("192.168.178.1", 8080)])
        t = self.tm.get_last_task()
        t2 = Task.from_json(t.to_json())
        assert t == t2

    def teardown_method(self) -> None:
        self.tm.delete_table()
