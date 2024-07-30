FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y build-essential && \
    apt-get install -y ffmpeg wget nginx ca-certificates && \
    apt-get install -y sox libsox-dev python3-dev python3-pip python3-distutils && \
    apt-get clean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Symlink python3.X to python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

WORKDIR /opt/program

COPY ./requirements.txt /opt/program/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /opt/program/requirements.txt
COPY /src /opt/program/src/
COPY entrypoint.sh /opt/program/

RUN chmod +x /opt/program/src/serve_app.py

ENTRYPOINT "/opt/program/entrypoint.sh"