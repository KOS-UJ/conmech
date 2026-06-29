FROM python:3.11-slim-buster

WORKDIR /usr/src/app

RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install -y git

RUN apt-get install -y python3-gmsh

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
RUN uv sync --group dev

ENV PYTHONPATH "${PYTHONPATH}:."
ENV PATH="/usr/src/app/.venv/bin:${PATH}"
