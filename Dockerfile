FROM tensorflow/tensorflow:latest-gpu

RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN apt update && \
    apt install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y python3.9 python3-pip python3-venv python3-dev python3.9-distutils nvidia-cuda-toolkit
docker

RUN pip install poetry

WORKDIR /app

COPY . .
RUN poetry install
EXPOSE 8000
# Run your app
COPY . /app
CMD [ "poetry", "run", "start" ]