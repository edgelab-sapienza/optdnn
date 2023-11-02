A Simple framework for Tensorflow model optimization  

## HOW TO RUN (with docker):
Step 1, build the container:

    docker build . -t optimizer-main

Step 2, run the container:

    docker run --gpus all -dp 192.168.0.2:8000:8000 optimized-main

Where the IP address have to be replaced with your IP address


