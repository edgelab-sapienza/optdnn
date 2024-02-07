import argparse
import asyncio
import logging
import multiprocessing
import os
import pathlib
import sys
import tempfile

import tensorflow as tf

from tf_optimizer.benchmarker.benchmarker import Benchmarker
from tf_optimizer.configuration import Configuration
from tf_optimizer.dataset_manager import DatasetManager
from tf_optimizer.optimizer.optimization_param import ModelProblemInt, QuantizationParameter, \
    QuantizationTechnique, QuantizationType
from tf_optimizer.optimizer.optimizer import Optimizer
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

    input_file = None
    dataset_path = None
    batch_size = None
    ds_scale = None
    img_size = None
    remote_addr = None
    remote_port = None
    configuration = Configuration()

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

    parser.add_argument(
        "--dataset_format",
        type=str,
        help="Dataset image format, [tf, torch, caffe]",
        required=False,
        choices=['tf', 'torch', 'caffe'],
        default=None,
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
    original: tf.keras.Sequential = tf.keras.models.load_model(input_file)

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

    # train, val = dm.generate_batched_dataset(batch_size=batch_size)

    tuner = Tuner(original, dm, ModelProblemInt.CATEGORICAL_CLASSIFICATION, batch_size)
    setup_logger('log_one', "LOG_ONE.log")
    logger(f"OPTIMIZING {os.path.basename(input_file)}", 'info', 'one')
    result = await tuner.test_model(input_file)
    print(f"MEASURED RESULTS {result}")
    original_acc = result.accuracy
    logger(f"ORIGINAL ACCURACY {original_acc}", 'info', 'one')
    model_path = tempfile.mktemp("*.keras")
    original.save(model_path)

    device = EdgeDevice("192.168.0.113", 22051)
    # device = EdgeDevice("192.168.0.68", 12300)
    device.id = 0

    bc = Benchmarker(edge_devices=[device])
    optimizer = Optimizer(dm, ModelProblemInt.CATEGORICAL_CLASSIFICATION, batch_size)

    qTypes: list[QuantizationType] = \
        [QuantizationType.ForceInt8, QuantizationType.WeightInt8ActivationInt16, QuantizationType.Standard,
         QuantizationType.AllFP16]
    # Test this quantization types in order, the first which matches the accuracy is used
    quantization_parameter = QuantizationParameter()
    quantization_parameter.set_quantization_technique(QuantizationTechnique.PostTrainingQuantization)
    for qType in qTypes:
        quantization_parameter.quantizationType = qType
        quantized_model: bytes = await optimizer.quantize_model(original, quantization_parameter)
        bc.add_tf_lite_model(quantized_model, qType.name)
    await bc.set_dataset(dm)
    bc.add_model(original, "Original")
    results = await bc.benchmark()
    for result in results["0"]:
        # print(f"NAME:{result.name}\tTIME:{result.time}\tSIZE:{result.size}\tACC:{result.accuracy}")
        logger(f"NAME:{result.name}\tTIME:{result.time}\tSIZE:{result.size}\tACC:{result.accuracy}", 'info', 'one')




if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    asyncio.run(main())
