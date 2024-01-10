import argparse
import asyncio
import multiprocessing
import pathlib
import sys
import tempfile

import tensorflow as tf

from tf_optimizer.benchmarker.benchmarker import Benchmarker
from tf_optimizer.dataset_manager import DatasetManager
from tf_optimizer.optimizer.optimization_param import ModelProblemInt
from tf_optimizer.optimizer.tuner import Tuner
from tf_optimizer.task_manager.edge_device import EdgeDevice
from tf_optimizer.task_manager.task import Task


async def main():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    gpu = gpus[0]
    tf.config.experimental.set_memory_growth(gpu, True)

    parser = argparse.ArgumentParser(
        prog="tf_optimizer",
        description="A tool for optimization of TF models deployed on TFLite devices",
    )
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        help=".h5 file representing the model",
        required="--remote" not in sys.argv,
    )

    parser.add_argument(
        "--dataset",
        type=pathlib.Path,
        help="path to the dataset",
        required="--remote" not in sys.argv,
    )

    parser.add_argument("--opt", type=int, help="Optimization")
    parser.add_argument(
        "--test_opt",
        action="store_true",
        help="Run single optimization with default param",
        default=False,
    )

    parser.add_argument(
        "--batch",
        type=int,
        help="Batch size (default 32)",
        required=False,
        default=32,
    )

    parser.add_argument(
        "--dataset_range",
        type=str,
        help="Dataset desired scale, min a max separated by a space, ex: --dataset_range 0 1",
        required=False,
        default=[0, 1],
        nargs=2,
    )

    parser.add_argument(
        "--image_size",
        type=str,
        help="Model input image size (default autodetected), ex: --image_size 250 250",
        required=False,
        default=[None, None],
        nargs=2,
    )

    args = parser.parse_args()

    input_file = args.input
    dataset_path = args.dataset
    optimizations = args.opt
    test_run = args.test_opt
    batch_size = args.batch
    ds_scale = args.dataset_range
    img_size = args.image_size

    ds_scale = list(map(lambda x: int(x), ds_scale))
    original = tf.keras.models.load_model(input_file)

    detected_input_size = original.input_shape
    if img_size[0] is not None and img_size[1] is not None:
        img_size[0] = int(img_size[0])
        img_size[1] = int(img_size[1])
        detected_input_size = (None, img_size[0], img_size[1], detected_input_size[3])
    if detected_input_size[1] is None and detected_input_size[2] is None:
        print(
            "Cannot detect model input size, provides it manually using the parameter --image_size"
        )
        exit()
    print(f"INPUT SIZE {detected_input_size}")
    img_shape = (detected_input_size[1], detected_input_size[2])
    # dm = DatasetManager(dataset_path, img_size=img_shape, scale=ds_scale)
    dm = DatasetManager(dataset_path, img_size=img_shape, data_format="caffe")
    tuner = Tuner(original, dm, ModelProblemInt.CATEGORICAL_CLASSIFICATION, batch_size)
    result = await tuner.test_model(input_file)
    print(f"MEASURED RESULTS {result}")
    original_acc = result.accuracy
    model_path = tempfile.mktemp("*.keras")
    original.save(model_path)

    device = EdgeDevice("192.168.0.68", 12300)
    device.id = 0
    bc = Benchmarker(edge_devices=[device])

    tuner = Tuner(original, dm, ModelProblemInt.CATEGORICAL_CLASSIFICATION, batch_size)
    tflite_model = await tuner.get_optimized_model()
    bc.add_tf_lite_model(tflite_model, "optimized")
    bc.add_model(original, "original")
    await bc.set_dataset(dm)
    results = await bc.benchmark()
    for result in results["0"]:
        print(f"NAME:{result.name}\tTIME:{result.time}\tSIZE:{result.size}\tACC:{result.accuracy}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    asyncio.run(main())
