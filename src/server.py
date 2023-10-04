from fastapi import FastAPI, Query
from typing_extensions import Annotated
from tf_optimizer.task_manager.optimization_config import OptimizationConfig
import uvicorn
from tf_optimizer.task_manager.task_manager import TaskManager
from tf_optimizer.task_manager.task import Task
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

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
tm = TaskManager()


@app.post("/add_task/", tags=["add_task"])
def add_task(optimization_config: OptimizationConfig):
    t = Task()
    t.model_url = str(optimization_config.model_url)
    t.dataset_url = str(optimization_config.dataset_url)
    t.dataset_scale = optimization_config.dataset_scale
    t.callback_url = str(optimization_config.callback_url)
    t.batch_size = optimization_config.batch_size
    t.img_size = optimization_config.img_size
    nodes = optimization_config.remote_nodes
    nodes = list(map(lambda x: (str(x[0]), x[1]), nodes))
    t.remote_nodes = nodes
    tm.add_task(t)
    return str(optimization_config)


@app.get("/get_tasks/", tags=["get_tasks"])
def get_tasks():
    all_tasks = tm.get_all_task()
    json_compatible_item_data = jsonable_encoder(all_tasks)
    return JSONResponse(content=json_compatible_item_data)


@app.get("/delete_task/", tags=["delete_task"])
def delete_task(task_id: Annotated[int, Query(description="Id of the task to remove")]):
    removed_tasks_number = tm.delete_task(task_id)
    if removed_tasks_number > 0:
        return JSONResponse({"success": True, "message": None})
    else:
        return JSONResponse({"success": False, "message": "Task not found"})


@app.get("/download_optimized_model/", tags=["download_opt_model"])
def download_model(
    task_id: Annotated[
        int, Query(description="Id of the task linked to the model to download")
    ]
):
    return "MISSING IMPLEMENTATION"


def start():
    """Launched with `poetry run start` at root level"""
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
