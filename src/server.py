import uvicorn
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from tf_optimizer.task_manager.optimization_config import OptimizationConfig
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
tm = TaskManager()
app = FastAPI(title="TF Optimizer", openapi_tags=tags_metadata)


@app.post("/add_task/", tags=["add_task"])
def add_task(optimization_config: OptimizationConfig):
    t = Task()
    t.model_url = str(optimization_config.model_url)
    t.dataset_url = str(optimization_config.dataset_url)
    t.dataset_scale = optimization_config.dataset_scale
    t.callback_url = str(optimization_config.callback_url)
    t.batch_size = optimization_config.batch_size
    t.img_size = optimization_config.img_size
    if optimization_config.remote_nodes is not None:
        nodes = optimization_config.remote_nodes
        nodes = list(map(lambda x: (str(x[0]), x[1]), nodes))
        t.remote_nodes = nodes
    tm.add_task(t)
    return str(optimization_config)


@app.get("/get_tasks/", tags=["get_tasks"])
def get_tasks():
    all_tasks = tm.get_all_task()
    json_compatible_item_data = jsonable_encoder(all_tasks)
    for e in json_compatible_item_data:
        e["status"] = TaskStatus(e["status"]).name
    return JSONResponse(content=json_compatible_item_data)


@app.get("/trigger/")
def trigger():
    tm.check_task_to_process()
    return "CIAO"


@app.get("/{task_id}/info", tags=["get_task"])
def get_task(task_id: int):
    task = tm.get_task_by_id(task_id)
    json_compatible_item_data = jsonable_encoder(task)
    return JSONResponse(content=json_compatible_item_data)


@app.get("/{task_id}/delete", tags=["delete_task"])
def delete_task(task_id: int):
    removed_tasks_number = tm.delete_task(task_id)
    if removed_tasks_number > 0:
        return JSONResponse({"success": True, "message": None})
    else:
        return JSONResponse({"success": False, "message": "Task not found"})


@app.get("/{task_id}/resume", tags=["resume_task"])
def resume_task(task_id: int):
    updated_rows = tm.update_task_state(task_id, TaskStatus.PENDING)
    if updated_rows > 0:
        return JSONResponse({"success": True, "message": None})
    else:
        return JSONResponse({"success": False, "message": "Task not found"})


@app.get("/{task_id}/download", tags=["download_opt_model"])
def download_model(task_id: int):
    return "MISSING IMPLEMENTATION"


@app.get("/{task_id}/stop", tags=["stop_task"])
def stop_task(task_id: int):
    tm.terminate_task(task_id)
    return JSONResponse({"success": True, "message": "Process will terminate in few minutes"})


def start():
    """Launched with `poetry run start` at root level"""
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
