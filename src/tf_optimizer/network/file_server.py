import hashlib
import os
import pathlib
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer

from tf_optimizer.benchmarker.utils import get_gzipped_file


class StateHTTPServer(HTTPServer):
    """
    HTTP Server that knows a certain filename and can be set to remember if
    that file has been transferred using :class:`FileHandler`
    """

    downloaded = False
    filename = ""
    allowed_basenames: list = []
    reporthook = None
    created_files_paths = []


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        hashed_basenames = map(
            lambda x: "/" + hashlib.sha1(x.encode()).hexdigest(),
            self.server.allowed_basenames,
        )
        if self.path in hashed_basenames:
            full_path = os.path.join(os.curdir, self.server.filename)
            if pathlib.Path(full_path).suffix == ".zip":
                zipped_path = full_path
            else:
                zipped_path = get_gzipped_file(full_path)
                self.server.created_files_paths.append(zipped_path)
            with open(zipped_path, "rb") as fh:
                maxsize = os.path.getsize(zipped_path)
                self.send_response(200)
                self.send_header("Content-type", "application/octet-stream")
                self.send_header(
                    "Content-disposition",
                    'inline; filename="%s"' % os.path.basename(zipped_path),
                )
                self.send_header("Content-length", maxsize)
                self.end_headers()

                i = 0
                while True:
                    data = fh.read(1024 * 8)  # chunksize taken from urllib
                    if not data:
                        break
                    self.wfile.write(data)
                    if self.server.reporthook is not None:
                        self.server.reporthook(i, 1024 * 8, maxsize)
                    i += 1

            self.server.downloaded = True
        else:
            self.send_header("Content-type", "html")
            self.send_response(404)
            self.end_headers()


class FileServer:
    def __init__(self, path: str) -> None:
        self.path = path
        self.port = int(os.environ.get("file_server_port", 8080))
        local_address = os.environ.get("file_server_ip_address", None)
        if local_address is None:
            self.ip = (
            (([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")] or [
                [(s.connect(("8.8.8.8", 53)), s.getsockname()[0], s.close()) for s in
                 [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) + ["no IP found"])[0])
        else:
            self.ip = local_address

    def get_file_url(self) -> str:
        filename = os.path.basename(self.path)
        file_hash = hashlib.sha1(filename.encode()).hexdigest()
        return f"http://{self.ip}:{self.port}/{file_hash}"

    def serve(self):
        with StateHTTPServer(("", self.port), handler) as server:
            server.allowed_basenames.append(os.path.basename(self.path))
            server.filename = self.path
            while True:
                server.handle_request()
                if server.downloaded is True:
                    # Remove allocated temp files
                    for path in server.created_files_paths:
                        if os.path.exists(path):
                            os.remove(path)
                    break
