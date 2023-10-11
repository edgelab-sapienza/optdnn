import os
import tempfile


class ModelInfo:
    size: int = 0
    time: float = 0
    accuracy: float = 0

    def __init__(self, model: bytes, name: str, is_reference=False) -> None:
        self.model_path = tempfile.mktemp(".tflite")
        with open(self.model_path, "wb") as f:
            f.write(model)
        self.name = name
        self.is_reference = is_reference

    def __str__(self):
        return f"{type(self).__name__} | {vars(self)}"

    def get_model(self) -> bytes:
        with open(self.model_path, "rb") as f:
            return f.read()

    def get_model_path(self):
        return self.model_path

    def __del__(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
