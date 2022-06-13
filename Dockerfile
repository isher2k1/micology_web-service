FROM python:3.9-slim

COPY . /root

WORKDIR /root

RUN pip3 install flask gunicorn tensorflow numpy flask_wtf pillow