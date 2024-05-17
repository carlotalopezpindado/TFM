FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Actualizar la lista de paquetes e instalar Python y otras utilidades necesarias
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-dev netcat-openbsd && \
    ln -s /usr/bin/python3 /usr/bin/python

# Instalar dependencias de Python
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

# Copiar archivos de la aplicación al contenedor
COPY config.ini /app/config.ini
COPY processing/indexes /app/processing/indexes
COPY app/main.py /app/main.py

# Establecer el directorio de trabajo
WORKDIR /app

# Comando modificado para esperar a MySQL y ejecutar main.py
CMD ["sh", "-c", "until nc -z mysql 3306; do echo waiting for mysql; sleep 1; done; python main.py"]