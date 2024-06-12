FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-dev netcat-openbsd && \
    ln -s /usr/bin/python3 /usr/bin/python

RUN pip3 install nvidia-ml-py3

COPY requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

COPY config.ini /app/config.ini
COPY processing/indexes /app/processing/indexes
COPY app/main.py /app/main.py

WORKDIR /app

CMD ["sh", "-c", "until nc -z mysql 3306; do echo waiting for mysql; sleep 1; done; streamlit run main.py"]