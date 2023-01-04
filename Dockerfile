FROM python:3.8-slim
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_22-01.html#rel_22-01
# This image is very large (14.8GB)
# FROM nvcr.io/nvidia/pytorch:22.01-py3

RUN rm -rf /workspace/*
WORKDIR /workspace/pytorch-ocr

COPY ./requirements.txt ./

RUN pip3 install --no-cache-dir --upgrade --pre pip
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .
