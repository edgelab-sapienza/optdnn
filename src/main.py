import argparse
import pathlib
from tf_optimizer.optimizer.optimizer import Optimizer
from tf_optimizer.optimizer.optimization_param import (
    OptimizationParam,
    QuantizationLayerToPrune,
)
from tf_optimizer.optimizer.tuner import Tuner
from tf_optimizer.benchmarker.benchmarker import Benchmarker, FieldToOrder
from tf_optimizer.network.client import Client
from tf_optimizer.dataset_manager import DatasetManager
from tf_optimizer.configuration import Configuration
import inquirer
import tensorflow as tf
import sys
import asyncio
import multiprocessing
from time import time


def get_optimizations() -> OptimizationParam:
    """
    Ask to the users the optimization that want to apply
    """
    questions = [
        inquirer.Checkbox(
            "optimization",
            message="What optimization do you want to apply?",
            choices=["pruning", "quantization", "clustering"],
        ),
    ]

    answers = inquirer.prompt(questions)
    optimization = OptimizationParam()
    if "pruning" in answers["optimization"]:
        optimization.toggle_pruning(True)
    if "quantization" in answers["optimization"]:
        optimization.toggle_quantization(True)
    if "clustering" in answers["optimization"]:
        optimization.toggle_clustering(True)
    return optimization


async def main():
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
        "--remote_addr",
        type=str,
        help="IP to the remote node used for testing",
        required=False,
    )

    __remote_port__ = configuration.getConfig("REMOTE", "port")
    parser.add_argument(
        "--remote_port",
        type=int,
        help="IP to the remote node used for testing (default {})".format(
            __remote_port__
        ),
        required=False,
        default=__remote_port__,
    )

    parser.add_argument(
        "--interface_addr",
        type=str,
        help="Address of local machine (default autodetected)",
        required=False,
        default=None,
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

    gpus = tf.config.experimental.list_physical_devices("GPU")
    gpu = gpus[0]
    tf.config.experimental.set_memory_growth(gpu, True)

    args = parser.parse_args()

    input_file = args.input
    dataset_path = args.dataset
    optimizations = args.opt
    test_run = args.test_opt
    batch_size = args.batch
    ds_scale = args.dataset_range
    img_size = args.image_size
    use_remote_nodes = args.remote_addr is not None
    if use_remote_nodes is True:
        client = Client(args.remote_addr, args.remote_port, args.interface_addr)
    else:
        client = None

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
    dm = DatasetManager(dataset_path, img_size=img_shape, scale=ds_scale)

    if test_run:
        bm = Benchmarker(
            use_remote_nodes=use_remote_nodes, client=client, use_multicore=False
        )
        # Ask for optimizations
        if optimizations is None:
            optimizations = get_optimizations()
        optimizations.set_quantized_layers(QuantizationLayerToPrune.AllLayers)
        optimizations.set_in_out_type(tf.uint8)
        optimizer = Optimizer(
            input_file,
            dataset_manager=dm,
            optimization_param=optimizations,
            batch_size=batch_size,
        )
        tf_lite_model = optimizer.optimize()
        original = tf.keras.models.load_model(input_file)
        bm.add_model(original, "Original")
        bm.add_tf_lite_model(tf_lite_model, "Optimized")

        await bm.set_dataset(dm)
        await bm.benchmark()
        bm.summary(fieldToOrder=FieldToOrder.Size)
        await bm.clear_online_node()
    else:
        start_time = time()
        tuner = Tuner(
            original,
            dm,
            use_remote_nodes=use_remote_nodes,
            client=client,
            batchsize=batch_size,
        )
        await tuner.tune()
        end_time = time()
        print(f"Tooked time {end_time-start_time} seconds")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    asyncio.run(main())
