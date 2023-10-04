from tf_optimizer.task_manager.task_manager import TaskManager
from tf_optimizer.task_manager.task import Task, TaskStatus


class TestClass:
    def setup_method(self) -> None:
        self.tm = TaskManager()

    def __add_task__(self) -> Task:
        t = Task()
        t.model_url = "URL DI PROVA"
        t.dataset_url = "URL DI PROVA DATASET"
        t.dataset_scale = "[-1,1]"
        t.callback_url = "URL CALLBACK"
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
        self.tm.update_task_state(inserted_task.id, TaskStatus.FAILED)
        updated_task = self.tm.get_task_by_id(inserted_task.id)
        assert updated_task.status == TaskStatus.FAILED

    def test_serialization(self) -> None:
        t1 = Task()
        t1.model_url = "URL DI PROVA"
        t1.dataset_url = "URL DI PROVA DATASET"
        t1.dataset_scale = "[-1,1]"
        t1.callback_url = "URL CALLBACK"
        t1.remote_nodes = [["192.168.178.3", 3], ["192.168.178.5", 565]]
        t1.img_size = [200, 200]
        t2 = Task.from_json(t1.to_json())        
        assert t1 == t2

    def teardown_method(self) -> None:
        self.tm.delete_table()
