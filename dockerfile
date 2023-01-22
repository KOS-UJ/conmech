FROM python:3.9.12-slim-buster

WORKDIR /usr/src/app

RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install -y git

RUN apt-get install -y python3-gmsh
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install pytest pytest-env pytest-xdist
RUN pip install -r requirements.txt
RUN rm requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:."
