import tensorflow as tf
import tempfile
import os


class ModelHandler:
    """
    A list in which models are saved in the disk a temporany files and not in memory
    """

    path_lists = []

    def add_model(self, model: tf.keras.Sequential):
        path = tempfile.mktemp(".keras")
        model.save(path)
        self.path_lists.append(path)

    def clear(self):
        for e in self.path_lists:
            os.remove(e)
        self.path_lists.clear()

    def get_model_by_index(self, index: int):
        return tf.keras.models.load_model(self.path_lists[index])

    def __del__(self):
        self.clear()
