FROM python:3.9.12-slim-buster

WORKDIR /usr/src/app

RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install -y python3-gmsh

COPY requirements.txt ./
RUN pip install pytest
RUN pip install -r requirements.txt

#COPY requirements-deep.txt ./
#RUN pip install torch>=1.11.0
##RUN pip install -r requirements-deep.txt

ENV PYTHONPATH "${PYTHONPATH}:."

# COPY . .

# CMD [ "python", "./run.py" ]