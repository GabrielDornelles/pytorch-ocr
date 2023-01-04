# Test version with cuda support for pytorch
FROM python:3.8-slim

RUN rm -rf /workspace/*
WORKDIR /workspace/pytorch-ocr

COPY ./requirements.txt ./

RUN pip3 install --no-cache-dir --upgrade --pre pip
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .