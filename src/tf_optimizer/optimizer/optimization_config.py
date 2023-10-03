from pydantic import BaseModel, Field, HttpUrl
from ipaddress import IPv4Address, IPv4Network
from typing import Union, Tuple
from fastapi import Query

NODE_ADDRESS_REGEX = "^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$:([\d]{1,5})"

class OptimizationConfig(BaseModel):
    # Class containing the information to perform the optimization of a model
    model_url:Union[HttpUrl,IPv4Address] = Query(example="http://modelhost.com/mymodel.keras", description="URL used to download the .keras model")
    dataset_url:Union[HttpUrl,IPv4Address] = Query(example="http://datasethost.com/myhost.zip", description="URL used to download the zip file of the dataset")
    dataset_scale: Tuple[int, int] = Field(examples=["[-1, 1]"], description="Dataset desired scale")
    img_size: Union[Tuple[int, int], None] = Field(example="[224, 224]", description="Model input image size (default autodetected)", default=None)

    # Must be list of devices
    remote_nodes: Union[None,list[str]] = Query(example=["192.168.178.2:12345", "192.168.178.3:12345", "192.168.178.4:12345"], description="IP addresses and port of the remote nodes used to test the model", pattern=NODE_ADDRESS_REGEX)

    batch_size: int = Field(examples=["32"], description="Batch size", default=32)

    def __str__(self) -> str:
        return (
            f"Model url: {self.model_url} \n"
            + f"Dataset url {self.dataset_url} \n"
            + f"Dataset scale {self.dataset_scale} \n"
        )
