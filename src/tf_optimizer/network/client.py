import os
import shutil
import sys

import websockets
from tf_optimizer_core.benchmarker_core import Result
from tf_optimizer_core.protocol import Protocol, PayloadMeans

from tf_optimizer.network.file_server import FileServer


class Client:
    def __init__(self, addr: str, port: int = 12300, machine_addr: str = None) -> None:
        self.dst_add = addr
        self.port = port
        self.local_address = machine_addr

    async def send_model(
            self, model_path: str, model_name: str
    ) -> Result:
        uri = "ws://{}:{}".format(self.dst_add, self.port)
        async with websockets.connect(uri, ping_interval=None) as websocket:
            fs = FileServer(model_path, local_address=self.local_address)
            url = fs.get_file_url()
            text_message = url + Protocol.string_delimiter + model_name
            msg = Protocol.build_put_model_file_request(text_message)
            print(f"FILE AT {model_path} served on URL: {msg}")
            await websocket.send(msg.to_bytes())
            print(f"Uploading: {model_name}")
            fs.serve()  # Blocking
            while True:
                msg = await websocket.recv()
                p_msg = Protocol.build_by_message(msg)

                if p_msg.cmd == PayloadMeans.Result:
                    return Protocol.get_evaulation_by_msg(p_msg)
                elif p_msg.cmd == PayloadMeans.ProgressUpdate:
                    payload = p_msg.payload.decode("utf-8")
                    print(f"\r{payload}", end="")
                    sys.stdout.flush()
                else:
                    return None

    async def send_dataset(self, dataset: str):
        base_name = "my_dataset"
        filename = shutil.make_archive(base_name, "zip", dataset)

        uri = "ws://{}:{}".format(self.dst_add, self.port)
        fs = FileServer(filename, local_address=self.local_address)
        async with websockets.connect(uri) as websocket:
            url = fs.get_file_url().encode("utf-8")
            msg = Protocol.build_put_dataset_file_request(url)
            print(f"DS URL {msg}")
            await websocket.send(msg.to_bytes())
            print("Uploading dataset")
            fs.serve()  # Blocking
            msg = await websocket.recv()
        os.remove(filename)

    async def close(self):
        uri = "ws://{}:{}".format(self.dst_add, self.port)
        async with websockets.connect(uri) as websocket:
            close_msg = Protocol(PayloadMeans.Close, b"")
            await websocket.send(close_msg.to_bytes())

    def identifier(self) -> str:
        return f"{self.dst_add}:{self.port}"
