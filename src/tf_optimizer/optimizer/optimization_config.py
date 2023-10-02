from pydantic import BaseModel, Field
from typing import Union


class OptimizationConfig(BaseModel):
    # Class containing the information to perform the optimization of a model
    model_url: str = Field(examples=["http://modelhost.com/mymodel.keras"], description="URL used to download the .keras model")
    dataset_url: str = Field(examples=["http://datasethost.com/myhost.zip"], description="URL used to download the zip file of the dataset")
    dataset_scale: list = Field(examples=["[-1, 1]"], description="Dataset desired scale")
    img_size: Union[list, None] = Field(examples=["[224, 224]"], description="Model input image size (default autodetected)", default=None)
    remote_addr: Union[str, None] = Field(examples=["192.168.1.3"], description="IP to the remote node used for testing", default=None)
    remote_port: Union[int, None] = Field(examples=["12345"], description="IP to the remote node used for testing", default=12300)
    batch_size: int = Field(examples=["32"], description="Batch size", default=32)

    def __str__(self) -> str:
        return (
            f"Model url: {self.model_url} \n"
            + f"Dataset url {self.dataset_url} \n"
            + f"Dataset scale {self.dataset_scale} \n"
        )
