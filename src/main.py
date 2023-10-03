import argparse
from tf_optimizer.optimizer.tuner import Tuner
from tf_optimizer.network.client import Client
from tf_optimizer.dataset_manager import DatasetManager
from tf_optimizer.task_manager.optimization_config import OptimizationConfig
import tensorflow as tf
import asyncio
import multiprocessing
from time import time
from fastapi import FastAPI, Query
from typing_extensions import Annotated
import uvicorn


tags_metadata = [
    {
        "name": "add_task",
        "description": "Upload a model to be optimized, a new task is created and will be processed asynchronously",
    },
    {
        "name": "get_tasks",
        "description": "Get all the tasks queued or completed",
    },
    {
        "name": "delete_task",
        "description": "Remove a task from the processing queue",
    },
    {
        "name": "download_opt_model",
        "description": "Download an optimized model",
    },
]

app = FastAPI(title="TF Optimizer", openapi_tags=tags_metadata)


@app.post("/add_task/", tags=["add_task"])
def optimize(optimization_config: OptimizationConfig):
    return str(optimization_config)


@app.get("/get_tasks/", tags=["get_tasks"])
def get_tasks():
    return "MISSING IMPLEMENTATION"


@app.get("/delete_task/", tags=["delete_task"])
def delete_task(task_id: Annotated[int, Query(description="Id of the task to remove")]):
    return f"MISSING IMPLEMENTATION {task_id}"


@app.get("/download_optimized_model/", tags=["download_opt_model"])
def download_model(
    task_id: Annotated[
        int, Query(description="Id of the task linked to the model to download")
    ]
):
    return "MISSING IMPLEMENTATION"


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


def start():
    """Launched with `poetry run start` at root level"""
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
