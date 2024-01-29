import os
import shutil
import sys
from typing import List, Union

import websockets
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import Mapped, relationship

from tf_optimizer import Base
from tf_optimizer.dataset_manager import DatasetManager
from tf_optimizer.network.file_server import FileServer
from tf_optimizer.task_manager.benchmark_result import BenchmarkResult
from tf_optimizer_core.benchmarker_core import Result
from tf_optimizer_core.protocol import Protocol, PayloadMeans


class EdgeDevice(Base):
    __tablename__ = "edge_result"

    id = Column(Integer, primary_key=True, index=True)
    alias = Column(String)
    ip_address = Column(String, nullable=False)
    port = Column(Integer, nullable=False)
    # inference_time = Column(JSON, default=None)
    task_id = Column(ForeignKey("tasks.id"))
    task: Mapped["Task"] = relationship(back_populates="devices")
    results: Mapped[List["BenchmarkResult"]] = relationship(
        back_populates="edge", lazy="joined"
    )

    def __init__(self, ip_address: str, port: int):
        self.ip_address = ip_address
        self.port = port
        self.results = []

    def __str__(self):
        return f"{self.id} IP: {self.ip_address} - {self.port} - TASK ID:{self.task_id}"

    def print_result(self):
        print(f"EDGE DEVICE HAS {len(self.results)} RESULTS")
        for result in self.results:
            print(f"IM {self.identifier()}, MODEL :{self.inference_time}")

    async def send_model(self, model_path: str, model_name: str) -> Union[Result, None]:

        uri = "ws://{}:{}".format(self.ip_address, self.port)
        print(f"SENDING {model_path}")
        async with websockets.connect(uri, ping_interval=None) as websocket:
            fs = FileServer(model_path)
            url = fs.get_file_url()
            text_message = url + Protocol.string_delimiter + model_name + Protocol.string_delimiter + str(self.id)
            msg = Protocol.build_put_model_file_request(text_message)
            await websocket.send(msg.to_bytes())
            print(f"Uploading: {model_name}")
            fs.serve()  # Blocking
            while True:
                msg = await websocket.recv()
                p_msg = Protocol.build_by_message(msg)

                if p_msg.cmd == PayloadMeans.Result:
                    print()
                    return Protocol.get_result_by_msg(p_msg)
                elif p_msg.cmd == PayloadMeans.ProgressUpdate:
                    payload = p_msg.payload.decode("utf-8")
                    print(f"\r{payload}", end="")
                    sys.stdout.flush()
                else:
                    return None

    async def send_dataset(self, dataset_manager: DatasetManager):
        base_name = "my_dataset"
        dataset = dataset_manager.get_validation_folder()
        filename = shutil.make_archive(base_name, "zip", dataset)

        uri = "ws://{}:{}".format(self.ip_address, self.port)
        fs = FileServer(filename)
        async with websockets.connect(uri) as websocket:
            # Send scale
            if dataset_manager.scale is not None and len(dataset_manager.scale)>0:
                content = f"{dataset_manager.scale[0]}{Protocol.string_delimiter}{dataset_manager.scale[1]}"
                msg = Protocol(PayloadMeans.DatasetScale, content.encode())
                await websocket.send(msg.to_bytes())

            # Send format
            if dataset_manager.data_format is None:
                content = ""
            else:
                content = f"{dataset_manager.data_format}"
            print(f"SENDING {content}")
            msg = Protocol(PayloadMeans.DataFormat, content.encode())
            await websocket.send(msg.to_bytes())

            # Send URL
            url = fs.get_file_url().encode("utf-8")
            msg = Protocol.build_put_dataset_file_request(url)
            print(f"DS URL {msg}")
            await websocket.send(msg.to_bytes())
            print("Uploading dataset")
            fs.serve()  # Blocking
            msg = await websocket.recv()

        if os.path.exists(filename):
            os.remove(filename)

    async def close(self):
        uri = "ws://{}:{}".format(self.ip_address, self.port)
        async with websockets.connect(uri) as websocket:
            close_msg = Protocol(PayloadMeans.Close, b"")
            await websocket.send(close_msg.to_bytes())

    def is_local_node(self):
        return (
                self.port == 0 and self.alias == "local" and self.ip_address == "localhost"
        )

    def identifier(self) -> str:
        return f"{self.ip_address}:{self.port}"
