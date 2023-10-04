import argparse
from tf_optimizer.optimizer.tuner import Tuner
from tf_optimizer.network.client import Client
from tf_optimizer.dataset_manager import DatasetManager
import tensorflow as tf
import asyncio
import multiprocessing
from time import time


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

    parser = argparse.ArgumentParser(
        prog="tf_optimizer",
        description="A tool for optimization of TF models deployed on TFLite devices",
    )

    parser.add_argument(
        "--interface_addr",
        type=str,
        help="Address of local machine (default autodetected)",
        required=False,
        default=None,
    )

    use_remote_nodes = remote_addr is not None
    args = parser.parse_args()
    if use_remote_nodes is True:
        client = Client(remote_addr, remote_port, args.interface_addr)
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
