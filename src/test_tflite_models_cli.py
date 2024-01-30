import argparse
import asyncio
import logging
import multiprocessing
import pathlib
import sys

from tf_optimizer.benchmarker.benchmarker import Benchmarker
from tf_optimizer.dataset_manager import DatasetManager
from tf_optimizer.task_manager.edge_device import EdgeDevice
from tf_optimizer.task_manager.task import Task


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
    parser = argparse.ArgumentParser(
        prog="tf_optimizer",
        description="A tool for optimization of TF models deployed on TFLite devices",
    )
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        help=".h5 file representing the model",
        required="--remote" not in sys.argv,
        nargs="+"
    )

    parser.add_argument(
        "--dataset",
        type=pathlib.Path,
        help="path to the dataset",
        required="--remote" not in sys.argv,
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
        "--edge_address",
        type=str,
        help="IP Address and port of edge node, ex: --edge_address 192.168.0.2:12300",
        required=True,
    )
    setup_logger('log_one', "TEST_MODELS.log")

    args = parser.parse_args()

    input_files = args.input
    dataset_path = args.dataset
    img_size = args.image_size
    ds_scale = args.dataset_range
    edge_address = args.edge_address

    img_size[0] = int(img_size[0])
    img_size[1] = int(img_size[1])
    detected_input_size = (None, img_size[0], img_size[1], 3)

    img_shape = (detected_input_size[1], detected_input_size[2])
    if args.dataset_format is not None:
        dm = DatasetManager(dataset_path, img_size=img_shape, data_format=args.dataset_format)
    else:
        dm = DatasetManager(dataset_path, img_size=img_shape, scale=ds_scale)

    if ":" not in edge_address:
        print("edge address not valid, use the format 192.168.0.2:12300")
        exit(-1)
    ip_address, port = edge_address.split(":")
    port = int(port)
    device = EdgeDevice(ip_address, port)
    device.id = 0

    bc = Benchmarker(edge_devices=[device])
    for model in input_files:
        m_str= str(model)
        f = open(m_str, mode="rb")
        data = f.read()
        bc.add_tf_lite_model(data, m_str)

    await bc.set_dataset(dm)
    results = await bc.benchmark()
    for result in results["0"]:
        # print(f"NAME:{result.name}\tTIME:{result.time}\tSIZE:{result.size}\tACC:{result.accuracy}")
        logger(f"NAME:{result.name}\tTIME:{result.time}\tSIZE:{result.size}\tACC:{result.accuracy}", 'info', 'one')


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    asyncio.run(main())
