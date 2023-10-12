import os
import zipfile
import tempfile


def get_gzipped_model_size(file):
    # Returns size of gzipped model, in bytes.
    zipped_file = get_gzipped_file(file)
    filesize = os.path.getsize(zipped_file)
    os.remove(zipped_file)
    return filesize


def get_tflite_model_size(model: bytes):
    temp_path = tempfile.mktemp(".tflite")
    with open(temp_path, "wb") as f:
        f.write(model)
        if os.path.exists(temp_path):
            size = get_gzipped_model_size(temp_path)
            os.remove(temp_path)
        else:
            size = 0
    return size


def get_gzipped_file(file):
    _, zipped_file = tempfile.mkstemp("_optimizer.zip")
    with zipfile.ZipFile(zipped_file, "w", compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file, os.path.basename(file))
    return zipped_file


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.2f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"
