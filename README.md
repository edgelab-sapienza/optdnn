# OptDNN: Automatic Deep Neural Networks Optimizer for Edge Computing 
 
**OptDNN is a automatic optimizer for neural network for edge computing.**
The software takes as inputs: a keras model, a dataset, and some other information, it springs in action its newel agortihms to return a TensorFlow Lite optimized model with an accuracy close to the one of the original model, but with a speed-up of the inference time and a smaller model size.
  
  ## HOW TO RUN (without docker): 
  The used dependency manager is [Poetry](https://python-poetry.org/), which can be installed with pip using this command:
  

    pip install poetry

  Now, from the project directory run:
  

    poetry install
To create a virtual environment in which all dependencies of the project will be installed.
Now you can run the software as HTTP server with:

    poetry run start
Or you can used the software with the CLI commands running:

    poetry shell
    python src/cli.py

## HOW TO RUN (with docker):
The Dockerfile actually allow to use the software only with the HTTP APIs.
A docker image already built is available in [docker hub](https://hub.docker.com/r/kernelmachine/optdnn) as `kernelmachine/optdnn`

Step 1, build the container:  

    docker build . -t optimizer

Step 2, run the container:  

    docker run --gpus all -dp 8000:8000 optimizer
  
Where the IP address have to be replaced with your IP address

### How to use HTTP APIs
The software, by default, will use the port 8000 for the HTTP APIs, and 8080 to serve the files to the remote nodes.
Once you have started the server, the documentation for the API is on:
http://127.0.0.1:8000/docs
Where IP address have to be replaced with the IP address of your host if accessed by another host.
### How to use CLI
The arguments for the CLI interface are:
- **`--input`** string containing the path to the model file to be optimized, the supported formats are .keras and SavedModels.\
ex.: `--input␣../models_generator/model.keras` [Required]
- **`--dataset`** string containing the path of a folder with the dataset, the path must contains several folders, each for each class, and each folder must contains the images, both jpg and png formats are supported.
The file structure should be compatible with: `tf.keras.utils.image_dataset_from_directory`\
ex.: `--dataset␣../models_generator/imagenet_dataset` [Required]
- **`--dataset_range`** pair of integers separated by a comma, they represents the minimum and the maximum value of the image values that the model expect as input.\
ex.: `--dataset_range -1, 1`,  mandatory if *--dataset_format* is not passed.
- **`--dataset_format`** one string between `{tf, torch, caffe}`, it represents the format of the input expected by the network, mandatory if *--dataset_range* is not passed.
More details [here](https://www.tensorflow.org/api_docs/python/tf/keras/applications/imagenet_utils/preprocess_input#args).
	-   **caffe**: will convert the images from RGB to BGR, then will zero-center each color channel with respect to the ImageNet dataset, without scaling.
	-   **tf**: will scale pixels between -1 and 1, sample-wise.
	-   **torch**: will scale pixels between 0 and 1 and then will normalize each channel with respect to the ImageNet dataset. Defaults to "caffe".

- **`--batch`** integer who represents the batch size, higher value is better, but can run out GPU memory.\
ex.: `--batch 32`. [Default 32]
- **`--image_size`** pair of integers, sometime is not possible to autodetect the
input shape for a given model, use this parameter to manually set the input shape.\
ex.: `--image_size 250, 250` [Default autodetected].
- **`--edge_addresses`** one or more IP addresses and ports separated by a space of the edge devices for testing the optimized model, if not passed, the local machine will be used also for the final evaluation.
ex.: `--edge_addresses 192.168.1.5:12300 192.168.1.6:12300` [Default None].
- **`--force_uint8`** if passed, the final model will be quantized to be run on only 8-bit devices.
- **`--model_problem`** the kind of problem solved by the model, one between:
  - categorical_classification
  - binary_classification
  
  ex.: `--model_problem binary_classification` [Default categorical_classification]

## Change default parameters
It is possible to change some parameters related to the optimization editing the files:
- .env
- config.ini

However the default parameters are fine for a large set of models.
