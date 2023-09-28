from tf_optimizer.benchmarker.model_info import ModelInfo


class Result:
    speedup = 0

    def __init__(self, model: ModelInfo, id=-1) -> None:
        self.model = model
        self.name = model.name
        self.time = model.time
        self.accuracy = model.accuracy
        self.size = model.size
        self.id = id
