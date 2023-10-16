import os
import tensorflow as tf
import tempfile
import shutil
import pickle
from random import randint


class DatasetManager:
    VALIDATION_SPLIT = 1 / 3

    def __init__(
        self,
        dataset_path,
        img_size,
        scale=[0, 1],
        random_seed: int = None,
        images_to_take=-1,
    ) -> None:
        self.dataset_path = dataset_path
        self.scale = scale
        self.img_size = img_size
        self.__files_dataset__ = None
        self.__validation_folder__ = None
        if random_seed is None:
            self.__random_seed__ = randint(1, 1000)
        else:
            self.__random_seed__ = random_seed
        self.__images_to_take__ = images_to_take

    def generate_batched_dataset(self, batch_size=32):
        interval_min = self.scale[0]
        interval_max = self.scale[1]
        interval_range = interval_max - interval_min

        def gen_element(filename):
            file = tf.io.read_file(filename)
            if tf.image.is_jpeg(file):
                image = tf.image.decode_jpeg(file)
            else:
                image = tf.image.decode_png(file)
            image = tf.image.resize(image, self.img_size)
            image = interval_min + (interval_range * tf.cast(image, tf.float32) / 255.0)
            return image

        def gen_label(filename):
            parts = tf.strings.split(filename, os.sep)
            label = parts[-2]
            return float(label)

        dataset = self.__generate_list_of_file__()
        files_number = int(dataset.cardinality())
        validation_size = int(files_number * self.VALIDATION_SPLIT)
        ds = dataset.map(lambda x: (gen_element(x), gen_label(x)))

        validation_ds = ds.take(validation_size).batch(batch_size)
        train_ds = ds.skip(validation_size).batch(batch_size)

        return (train_ds, validation_ds)

    def __generate_list_of_file__(self):
        if self.__files_dataset__ is None:
            all_files = os.path.join(os.path.join(self.dataset_path, "*"), "*")

            # It is run only once
            self.__files_dataset__ = tf.data.Dataset.list_files(
                all_files,
                shuffle=False,
            )
            self.__files_dataset__ = self.__files_dataset__.shuffle(
                buffer_size=self.__files_dataset__.cardinality(),
                reshuffle_each_iteration=False,
                seed=self.__random_seed__,
            )
            if self.__images_to_take__ > 0:
                self.__files_dataset__ = self.__files_dataset__.take(
                    self.__images_to_take__
                )
        return self.__files_dataset__

    def get_validation_folder(self) -> str:
        if self.__validation_folder__ is not None:
            return self.__validation_folder__
        ds = self.__generate_list_of_file__()
        ds_size = int(ds.cardinality())
        validation_ds = ds.take(int(ds_size * self.VALIDATION_SPLIT))
        temp_folder = tempfile.mkdtemp()
        for element in validation_ds.as_numpy_iterator():
            element = element.decode("utf-8")
            dst_path = element.split(os.sep)[-2:]
            class_folder = dst_path[-2]
            dst_path = os.sep.join(dst_path)
            os.makedirs(os.path.join(temp_folder, class_folder), exist_ok=True)
            shutil.copy(element, os.path.join(temp_folder, dst_path))
        self.__validation_folder__ = temp_folder
        return temp_folder

    def get_path(self) -> str:
        return self.dataset_path

    def __del__(self):
        if self.__validation_folder__ is not None:
            shutil.rmtree(self.__validation_folder__)

    def toJSON(self) -> bytes:
        d = {}
        d["dataset_path"] = self.dataset_path
        d["img_size"] = self.img_size
        d["scale"] = self.scale
        d["seed"] = self.__random_seed__
        d["images_to_take"] = self.__images_to_take__
        return pickle.dumps(d)

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, DatasetManager):
            dm: DatasetManager = __value
            return (
                self.dataset_path == dm.dataset_path
                and self.img_size == dm.img_size
                and self.scale == dm.scale
                and self.__random_seed__ == dm.__random_seed__
            )
        else:
            return False

    @staticmethod
    def fromJSON(jsonData: bytes):
        data = pickle.loads(jsonData)
        return DatasetManager(
            data["dataset_path"],
            data["img_size"],
            data["scale"],
            random_seed=data["seed"],
            images_to_take=data["images_to_take"],
        )
