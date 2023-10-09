from tf_optimizer.dataset_manager import DatasetManager
import os
import numpy as np
import pytest
import tensorflow as tf
import gdown
import zipfile


pytest_plugins = ("pytest_asyncio",)
dataset_path = "src/tests/resources/imagenet_dataset"
model_path = "src/tests/resources/test.h5"


class TestClass:
    images_to_take = -1

    @pytest.fixture
    def resource(self, request):
        if not os.path.exists(dataset_path):
            dataset_id = "1ejgS9pybY_nCy9DpZFlGPeYFPoxZq4II"
            dataset_out = "test_dataset.zip"
            gdown.download(id=dataset_id, output=dataset_out)
            with zipfile.ZipFile(dataset_out, "r") as zip_ref:
                zip_ref.extractall("tests/resources")
            os.remove(dataset_out)

        if not os.path.exists(model_path):
            model_id = "110RK_DF-hlpPBpSh0_rAk7y4UrffkIMY"
            model_out = "test_model.zip"
            gdown.download(id=model_id, output=model_out)
            with zipfile.ZipFile(model_out, "r") as zip_ref:
                zip_ref.extractall("tests/resources")
            os.remove(model_out)

        self.images_to_take = request.config.getoption("totake")

    def test_make_archive(self, resource):
        import glob

        dm = DatasetManager(dataset_path, img_size=(128, 128))
        print(dm.toJSON())
        path = dm.get_validation_folder()
        (_, val) = dm.generate_batched_dataset()

        val_size = sum(val.unbatch().map(lambda x, y: 1))
        all_files = os.path.join(os.path.join(path, "*"), "*")
        files_number = len(glob.glob(all_files))
        assert val_size == files_number and val_size > 0

    def test_dm_serialization_content(self, resource):
        dm = DatasetManager(dataset_path, img_size=(128, 128))
        serial = dm.toJSON()
        ele = DatasetManager.fromJSON(serial)
        vd1 = dm.generate_batched_dataset()[1].as_numpy_iterator()
        vd2 = ele.generate_batched_dataset()[1].as_numpy_iterator()
        batch1 = vd1.next()
        batch2 = vd2.next()
        assert len(batch1) == len(batch2)
        all_equal = False
        for i in range(len(batch1)):
            if np.array_equal(batch1[i], batch2[i]):
                all_equal = True
            else:
                all_equal = False
                break

        assert all_equal
        assert ele == dm

    def test_dm_serialization_files(self, resource):
        import filecmp
        import os.path

        def are_dir_trees_equal(dir1, dir2):
            """
            Compare two directories recursively. Files in each directory are
            assumed to be equal if their names and contents are equal.

            @param dir1: First directory path
            @param dir2: Second directory path

            @return: True if the directory trees are the same and
                there were no errors while accessing the directories or files,
                False otherwise.
            """

            dirs_cmp = filecmp.dircmp(dir1, dir2)
            if (
                len(dirs_cmp.left_only) > 0
                or len(dirs_cmp.right_only) > 0
                or len(dirs_cmp.funny_files) > 0
            ):
                return False
            (_, mismatch, errors) = filecmp.cmpfiles(
                dir1, dir2, dirs_cmp.common_files, shallow=False
            )
            if len(mismatch) > 0 or len(errors) > 0:
                return False
            for common_dir in dirs_cmp.common_dirs:
                new_dir1 = os.path.join(dir1, common_dir)
                new_dir2 = os.path.join(dir2, common_dir)
                if not are_dir_trees_equal(new_dir1, new_dir2):
                    return False
            return True

        original_dm = DatasetManager(dataset_path, img_size=(128, 128))
        serial = original_dm.toJSON()
        new_dm = DatasetManager.fromJSON(serial)
        original_validation_folder = original_dm.get_validation_folder()
        new_validation_folder = new_dm.get_validation_folder()

        assert are_dir_trees_equal(original_validation_folder, new_validation_folder)

    def test_optimization_param_serialization(self):
        from tf_optimizer.optimizer.optimization_param import OptimizationParam

        op = OptimizationParam()
        op.set_quantization_type(tf.int8)
        serial = op.toJSON()
        deserialized = OptimizationParam.fromJSON(serial)
        assert op == deserialized

    @pytest.mark.skipif("not config.getoption('longrun')")
    def test_comparison_tflite_and_keras(self, resource):
        m: tf.keras.Sequential = tf.keras.models.load_model(model_path)
        input_size = m.input_shape
        img_shape = (input_size[1], input_size[2])
        original_dm = DatasetManager(
            dataset_path,
            img_size=img_shape,
            scale=[-1, 1],
            images_to_take=self.images_to_take,
        )
        images_iter = (
            original_dm.generate_batched_dataset(2)[1].unbatch().as_numpy_iterator()
        )

        converter = tf.lite.TFLiteConverter.from_keras_model(m)
        tflite_model = converter.convert()

        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        input_index = input_details["index"]
        output_index = interpreter.get_output_details()[0]["index"]

        correct = 0
        total = 0
        tflite_correct_prediction = 0
        keras_correct_prediction = 0
        while True:
            img, label = next(images_iter, (None, None))
            if img is None or label is None:
                break
            img = np.expand_dims(img, axis=0)
            interpreter.set_tensor(input_index, img)
            interpreter.invoke()
            output_tflite = interpreter.get_tensor(output_index)
            tflite_prediction = output_tflite[0]
            keras_prediction = m(img)
            keras_prediction = keras_prediction[0]
            if np.argmax(tflite_prediction) == np.argmax(keras_prediction):
                correct += 1
            if np.argmax(keras_prediction) == label:
                keras_correct_prediction += 1
            if np.argmax(tflite_prediction) == label:
                tflite_correct_prediction += 1
            total += 1
        assert (
            correct == total and keras_correct_prediction == tflite_correct_prediction
        )

    @staticmethod
    def compare_tf_dataset_with_folder(
        tf_dataset: tf.data.Dataset, folder_path: str, scale=[0, 1], img_size=(128, 128)
    ) -> bool:
        def count_files(dir: str) -> int:
            count = 0
            for root_dir, cur_dir, files in os.walk(dir):
                count += len(files)
            return count

        def list_of_files(main_path: str):
            paths = []
            for folder in os.listdir(main_path):
                folder_path = os.path.join(main_path, folder)
                for img in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, img)
                    if os.path.isfile(img_path):
                        paths.append(img_path)
            return paths

        def open_img(path):
            file = tf.io.read_file(path)
            if tf.image.is_jpeg(file):
                image = tf.image.decode_jpeg(file)
            else:
                image = tf.image.decode_png(file)
            image = tf.image.resize(image, img_size)
            image = scale[0] + (
                (scale[1] - scale[0]) * tf.cast(image, tf.float32) / 255.0
            )
            return image

        validation_ds_size = sum(tf_dataset.map(lambda _, __: 1))
        validation_folder_size = count_files(folder_path)
        assert validation_ds_size == validation_folder_size  # Assert number of images

        imgs_files_paths = list_of_files(folder_path)
        imgs_from_dataset = tf_dataset.map(lambda x, y: x)
        equal_images = 0
        for img_from_file in map(lambda x: open_img(x), imgs_files_paths):
            img_from_dataset_iter = imgs_from_dataset.as_numpy_iterator()
            while True:
                img_from_dataset = next(img_from_dataset_iter, None)
                if img_from_dataset is None:
                    break
                if np.array_equal(img_from_file, img_from_dataset):
                    equal_images += 1
                    break

        print(f" {equal_images} of {validation_ds_size} are equals")
        return equal_images == validation_ds_size

    @pytest.mark.skipif("not config.getoption('longrun')")
    def test_validation_folder_with_dataset(self, resource):
        original_dm = DatasetManager(
            dataset_path,
            img_size=(128, 128),
            scale=[-1, 1],
            images_to_take=self.images_to_take,
        )
        _, validation_ds = original_dm.generate_batched_dataset(2)
        validation_ds = validation_ds.unbatch()
        validation_folder = original_dm.get_validation_folder()

        assert TestClass.compare_tf_dataset_with_folder(
            validation_ds, validation_folder, scale=original_dm.scale
        )

    @pytest.mark.skipif("not config.getoption('longrun')")
    @pytest.mark.asyncio
    async def test_accuracy_same_dm(self, resource):
        m = tf.keras.models.load_model(model_path)
        input_shape = (m.input_shape[1], m.input_shape[2])
        original_dm = DatasetManager(
            dataset_path,
            img_size=input_shape,
            scale=[-1, 1],
            images_to_take=self.images_to_take,
        )
        _, validation_ds = original_dm.generate_batched_dataset(2)
        validation_ds = validation_ds.unbatch()
        validation_folder = original_dm.get_validation_folder()

        converter = tf.lite.TFLiteConverter.from_keras_model(m)
        tflite_model = converter.convert()

        from tf_optimizer_core.benchmarker_core import BenchmarkerCore

        metrics = m.evaluate(validation_ds.batch(8))
        bc = BenchmarkerCore(validation_folder, original_dm.scale)
        result = await bc.test_model(tflite_model)
        assert np.isclose(metrics[1], result.accuracy)
        assert metrics[1] > 0

    @pytest.mark.skipif("not config.getoption('longrun')")
    @pytest.mark.asyncio
    async def test_accuracy_different_dm(self, resource):
        m = tf.keras.models.load_model(model_path)
        input_shape = (m.input_shape[1], m.input_shape[2])
        original_dm = DatasetManager(
            dataset_path,
            img_size=input_shape,
            scale=[-1, 1],
            images_to_take=self.images_to_take,
        )
        _, validation_ds = original_dm.generate_batched_dataset(2)
        validation_ds = validation_ds.unbatch()
        new_dataset_manager = DatasetManager.fromJSON(original_dm.toJSON())
        validation_folder = new_dataset_manager.get_validation_folder()

        converter = tf.lite.TFLiteConverter.from_keras_model(m)
        tflite_model = converter.convert()

        from tf_optimizer_core.benchmarker_core import BenchmarkerCore

        metrics = m.evaluate(validation_ds.batch(8))
        bc = BenchmarkerCore(validation_folder, new_dataset_manager.scale)
        result = await bc.test_model(tflite_model)
        assert np.isclose(metrics[1], result.accuracy)
        assert metrics[1] > 0

    @pytest.mark.skipif("not config.getoption('longrun')")
    @pytest.mark.asyncio
    async def test_framework_convertion(self, resource):
        from tf_optimizer.optimizer.optimizer import Optimizer
        from tf_optimizer.optimizer.optimization_param import OptimizationParam
        from tf_optimizer_core.benchmarker_core import BenchmarkerCore

        m = tf.keras.models.load_model(model_path)
        input_shape = (m.input_shape[1], m.input_shape[2])
        original_dm = DatasetManager(
            dataset_path,
            img_size=input_shape,
            scale=[-1, 1],
            images_to_take=self.images_to_take,
        )
        optimization_param = OptimizationParam()
        optimization_param.toggle_clustering(False)
        optimization_param.toggle_pruning(False)
        optimization_param.toggle_quantization(False)
        optimizer = Optimizer(model_path, original_dm, optimization_param, batch_size=8)

        _, validation_ds = original_dm.generate_batched_dataset(2)
        metrics = m.evaluate(validation_ds)

        bc = BenchmarkerCore(original_dm.get_validation_folder(), original_dm.scale)
        result = await bc.test_model(optimizer.optimize())
        assert np.isclose(metrics[1], result.accuracy)
        assert metrics[1] > 0
