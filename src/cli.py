import argparse
import asyncio
import logging
import multiprocessing
import os.path
import pathlib
import sys
import tempfile

import tensorflow as tf

from tf_optimizer.benchmarker.benchmarker import Benchmarker
from tf_optimizer.dataset_manager import DatasetManager
from tf_optimizer.optimizer.optimization_param import ModelProblemInt
from tf_optimizer.optimizer.tuner import Tuner
from tf_optimizer.task_manager.edge_device import EdgeDevice


def setup_logger(logger_name, log_file, level=logging.INFO):
    log_setup = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(levelname)s: %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    log_setup.setLevel(level)
    log_setup.addHandler(fileHandler)
    log_setup.addHandler(streamHandler)


def logger(msg, level, logfile):
    if logfile == 'one': log = logging.getLogger('log_one')
    if logfile == 'two': log = logging.getLogger('log_two')
    if level == 'info': log.info(msg)
    if level == 'warning': log.warning(msg)
    if level == 'error': log.error(msg)


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
        "--dataset_format",
        type=str,
        help="Dataset image format, [tf, torch, caffe]",
        required=False,
        choices=['tf', 'torch', 'caffe'],
        default=None,
    )
    parser.add_argument(
        "--image_size",
        type=str,
        help="Model input image size (default autodetected), ex: --image_size 250 250",
        required=False,
        default=[None, None],
        nargs=2,
    )
    parser.add_argument(
        "--edge_addresses",
        type=str,
        help="IP Address and port of edge node, ex: --edge_address 192.168.0.2:12300",
        required=False,
        default=[],
        nargs="*"
    )

    parser.add_argument(
        "--force_uint8",
        action="store_true",
        help="Force final quantization to be uint8, ex: --force_uint8",
        required=False,
    )

    model_problem_default = ModelProblemInt.CATEGORICAL_CLASSIFICATION.name.lower()
    parser.add_argument(
        "--model_problem",
        type=str,
        help=f"Kind of problem solved by the model, default: {model_problem_default}",
        required=False,
        default=model_problem_default,
        choices=[e.name.lower() for e in list(ModelProblemInt)],
    )

    args = parser.parse_args()

    input_file = args.input
    dataset_path = args.dataset
    batch_size = args.batch
    ds_scale = args.dataset_range
    img_size = args.image_size
    edge_addresses = args.edge_addresses
    force_uint8 = args.force_uint8

    setup_logger('log_one', "cli_log.log")
    logger(f"OPTIMIZING {input_file}", 'info', 'one')

    ds_scale = list(map(lambda x: int(x), ds_scale))
    original = tf.keras.models.load_model(input_file)

    model_problem = ModelProblemInt[args.model_problem.upper()]

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
    if args.dataset_format is not None:
        dm = DatasetManager(dataset_path, img_size=img_shape, data_format=args.dataset_format)
    else:
        dm = DatasetManager(dataset_path, img_size=img_shape, scale=ds_scale)
    tuner = Tuner(original, dm, model_problem, batch_size)
    tuner.force_uint8 = force_uint8
    result = await tuner.test_model(input_file)
    print(f"MEASURED RESULTS {result}")
    model_path = tempfile.mktemp("*.keras")
    original.save(model_path)

    devices = []
    id = 0
    for address in edge_addresses:
        if ":" not in address:
            print("edge address not valid, use the format 192.168.0.2:12300")
            exit(-1)
        ip_address, port = address.split(":")
        port = int(port)
        d = EdgeDevice(ip_address, port)
        d.id = id
        devices.append(d)
        id += 1
    bc = Benchmarker(edge_devices=devices)

    tflite_model = await tuner.get_optimized_model()
    bc.add_tf_lite_model(tflite_model, "optimized")
    bc.add_model(original, "original")
    await bc.set_dataset(dm)
    results = await bc.benchmark()
    for result in results["0"]:
        logger(f"NAME:{result.name}\tTIME:{result.time}\tSIZE:{result.size}\tACC:{result.accuracy}", 'info', 'one')

    output_path = "optimized_model"
    os.makedirs(output_path, exist_ok=True)
    original_file_name = os.path.basename(input_file)
    output_model_path = output_path+os.path.sep+f"{original_file_name}.tflite"
    with open(output_model_path, "wb") as f:
        f.write(tflite_model)
        f.close()
    print(f"MODEL SAVED IN: {output_model_path}")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    asyncio.run(main())
