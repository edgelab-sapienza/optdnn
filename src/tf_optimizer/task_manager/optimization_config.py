from enum import Enum
from ipaddress import IPv4Address
from typing import Union, Tuple

from fastapi import Query
from pydantic import BaseModel, Field
from pydantic.networks import HttpUrl

class DataFormat(str, Enum):
    tf = "tf"
    caffe = "caffe"
    torch = "torch"


class ModelProblem(str, Enum):
    CATEGORICAL_CLASSIFICATION = "categorical_classification"
    BINARY_CLASSIFICATION = "binary_classification"


class OptimizationConfig(BaseModel):
    # Class containing the information to perform the optimization of a model
    model_url: Union[HttpUrl, IPv4Address] = Query(example="http://modelhost.com/mymodel.keras",
                                                   description="URL used to download the .keras model")
    dataset_url: Union[HttpUrl, IPv4Address] = Query(example="http://datasethost.com/myhost.zip",
                                                     description="URL used to download the .zip file of the dataset")
    dataset_scale: Tuple[float, float] = Field(example=[-1, 1], description="Dataset desired scale", default=None)
    img_size: Union[Tuple[int, int], None] = Field(example=[224, 224],
                                                   description="Model input image size (default autodetected)",
                                                   default=None)

    # Must be list of devices
    remote_nodes: Union[None, list[Tuple[IPv4Address, int]]] = \
        Field(example=[["192.168.178.2", 12345], ["192.168.178.65", 12345], ["192.168.178.96", 12345]],
              description="IP addresses and port of the remote nodes used to test the model", default=None)

    callback_url: Union[HttpUrl, IPv4Address, None] = Query(
        example=["http://192.168.178.3:8080/callback?id=3", "http://my_pc.com/callback"],
        description="URL called when the optimization is ended", default=None)
    batch_size: int = Field(example=[32], description="Batch size", default=32)
    model_problem: ModelProblem = Field(example=ModelProblem.CATEGORICAL_CLASSIFICATION,
                                        description="Type of model problem")
    data_format: DataFormat = Field(example=DataFormat.tf, description="Type of data format", default=None)
    force_uint8: bool = Field(description="Force model to be quantized in int8, usefull for int8 devices", default=False)

    def __str__(self) -> str:
        return (
                f"Model url: {self.model_url} \n"
                + f"Dataset url {self.dataset_url} \n"
                + f"Dataset scale {self.dataset_scale} \n"
                + f"Data format {self.data_format} \n"
        )
