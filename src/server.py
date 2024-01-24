import multiprocessing

import uvicorn
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, FileResponse

from tf_optimizer.optimizer.optimization_param import ModelProblemInt
from tf_optimizer.task_manager.optimization_config import OptimizationConfig, ModelProblem
from tf_optimizer.task_manager.task import Task, TaskStatus
from tf_optimizer.task_manager.task_manager import TaskManager

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
        "name": "get_task",
        "description": "Get information of a single task",
    },
    {
        "name": "delete_task",
        "description": "Remove a task from the processing queue",
    },
    {
        "name": "resume_task",
        "description": "Resume a task to the pending status",
    },
    {
        "name": "stop_task",
        "description": "Kill a task",
    },
    {
        "name": "download_opt_model",
        "description": "Download an optimized model",
    },
]

# multiprocessing.set_start_method("spawn")
tm = TaskManager(run_tasks=True)
app = FastAPI(title="TF Optimizer", openapi_tags=tags_metadata)


@app.post(
    "/add_task/",
    tags=["add_task"],
    responses={
        200: {
            "description": "Inserted item",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "task": {
                            "dataset_url": "http://127.0.0.1:9000/fashion.zip",
                            "created_at": "2023-10-09T09:48:34.108757",
                            "img_size": None,
                            "callback_url": "http://127.0.0.1:9000",
                            "pid": None,
                            "model_url": "http://127.0.0.1:9000/fashion_mnist.keras",
                            "id": 2,
                            "status": 1,
                            "dataset_scale": [0, 1],
                            "remote_nodes": None,
                            "batch_size": 32,
                            "download_url_callback": "http://127.0.0.1:8000/2/download",
                        },
                    }
                }
            },
        }
    },
)
def add_task(optimization_config: OptimizationConfig, request: Request):
    t = Task()
    t.model_url = str(optimization_config.model_url)
    t.dataset_url = str(optimization_config.dataset_url)
    t.dataset_scale = optimization_config.dataset_scale
    t.callback_url = str(optimization_config.callback_url)
    t.batch_size = optimization_config.batch_size
    t.img_size = optimization_config.img_size

    print(optimization_config.model_problem)
    if optimization_config.model_problem is ModelProblem.CATEGORICAL_CLASSIFICATION:
        t.model_problem = ModelProblemInt.CATEGORICAL_CLASSIFICATION
    else:
        t.model_problem = ModelProblemInt.BINARY_CLASSIFICATION

    t.data_format = optimization_config.data_format

    remote_nodes = []
    if optimization_config.remote_nodes is not None:
        nodes = optimization_config.remote_nodes
        for node in nodes:
            ip_address = str(node[0])
            port = node[1]
            remote_nodes.append((ip_address, port))
    tm.add_task(t, base_url=request.base_url._url, nodes=remote_nodes)
    return JSONResponse({"success": True, "task": jsonable_encoder(t)})


@app.get(
    "/get_tasks/",
    tags=["get_tasks"],
    responses={
        200: {
            "description": "Get items",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "dataset_url": "http://127.0.0.1:9000/fashion.zip",
                            "created_at": "2023-10-06T22:00:05.817745",
                            "img_size": None,
                            "callback_url": "http://127.0.0.1:9000",
                            "pid": 270511,
                            "model_url": "http://127.0.0.1:9000/fashion_mnist.keras",
                            "id": 1,
                            "status": "COMPLETED",
                            "dataset_scale": [0, 1],
                            "remote_nodes": None,
                            "batch_size": 32,
                            "download_url_callback": "http://192.168.178.2:8000/1/download",
                        }
                    ]
                }
            },
        }
    },
)
def get_tasks():
    all_tasks = tm.get_all_task()
    json_compatible_item_data = jsonable_encoder(all_tasks)
    for e in json_compatible_item_data:
        e["status"] = TaskStatus(e["status"]).name
    return JSONResponse(content=json_compatible_item_data)


@app.get(
    "/{task_id}/info",
    tags=["get_task"],
    responses={
        200: {
            "description": "Get item",
            "content": {
                "application/json": {
                    "example": {
                        "dataset_url": "http://127.0.0.1:9000/fashion.zip",
                        "created_at": "2023-10-06T22:00:05.817745",
                        "img_size": None,
                        "callback_url": "http://127.0.0.1:9000",
                        "pid": 270511,
                        "model_url": "http://127.0.0.1:9000/fashion_mnist.keras",
                        "id": 1,
                        "status": 2,
                        "dataset_scale": [0, 1],
                        "remote_nodes": None,
                        "batch_size": 32,
                        "download_url_callback": "http://192.168.178.2:8000/1/download",
                    }
                }
            },
        }
    },
)
def get_task(task_id: int):
    task = tm.get_task_by_id(task_id)
    task["status"] = TaskStatus(task["status"]).name
    task["optimization_priority"] = OptimizationPriority(task["optimization_priority"]).name
    json_compatible_item_data = jsonable_encoder(task)
    return JSONResponse(content=json_compatible_item_data)


@app.get(
    "/{task_id}/delete",
    tags=["delete_task"],
    responses={
        200: {
            "description": "Task deleted",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "message": None,
                    }
                }
            },
        },
        400: {
            "description": "Task not found",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "message": "Task not found",
                    }
                }
            },
        },
    },
)
def delete_task(task_id: int):
    removed_tasks_number = tm.delete_task(task_id)
    if removed_tasks_number > 0:
        return JSONResponse({"success": True, "message": None})
    else:
        return JSONResponse(
            status_code=404, content={"success": False, "message": "Task not found"}
        )


@app.get(
    "/{task_id}/resume",
    tags=["resume_task"],
    responses={
        200: {
            "description": "Task resumed",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "message": None,
                    }
                }
            },
        },
        400: {
            "description": "Task not found",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "message": "Task not found",
                    }
                }
            },
        },
    },
)
def resume_task(task_id: int):
    updated_rows = tm.update_task_state(task_id, TaskStatus.PENDING)
    tm.update_task_field(task_id, "error_msg", None)
    tm.remove_results(task_id)
    if updated_rows > 0:
        return JSONResponse({"success": True, "message": None})
    else:
        return JSONResponse(
            status_code=404, content={"success": False, "message": "Task not found"}
        )


@app.get(
    "/{task_id}/download",
    tags=["download_opt_model"],
    name="download",
    responses={
        400: {
            "description": "Task not completed",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "message": "Task not completed",
                    }
                }
            },
        },
        404: {
            "description": "Task not found",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "message": "Task not found",
                    }
                }
            },
        },
        200: {
            "description": "Optimized model file",
            "content": {"application/octet-stream": {}},
        },
    },
)
def download_model(task_id: int):
    task = tm.get_task_by_id(task_id)
    if task is None:
        return JSONResponse(
            status_code=404, content={"success": False, "message": "Task not found"}
        )
    elif task.status != TaskStatus.COMPLETED:
        return JSONResponse({"success": False, "message": "Task not completed"})
    else:
        return FileResponse(
            task.generate_filename(),
            media_type="application/octet-stream",
            filename=f"optimized_model_t{task.id}.tflite",
        )


@app.get("/{task_id}/stop", tags=["stop_task"])
def stop_task(task_id: int):
    tm.terminate_task(task_id)
    return JSONResponse(
        {"success": True, "message": "Process will terminate in few minutes"}
    )


def start():
    """Launched with `poetry run start` at root level"""
    multiprocessing.set_start_method("spawn")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    start()
