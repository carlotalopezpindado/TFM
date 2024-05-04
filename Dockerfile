FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Instalar las dependencias necesarias
RUN pip install mysql-connector-python

# Actualizar la lista de paquetes y instalar netcat-openbsd
RUN apt-get update && apt-get install -y python3-pip python3-dev && ln -s /usr/bin/python3 /usr/bin/python
RUN apt-get update && apt-get install -y netcat-openbsd

# poner todas las librerias que sea necesario para los scripts
RUN pip install tensorflow-gpu

# Copiar solo el fichero config.ini y el directorio processing al directorio /app en el contenedor
COPY config.ini /app/config.ini
COPY processing/ /app/

# Establecer el directorio de trabajo
WORKDIR /app

# Comando modificado para esperar a MySQL
CMD ["sh", "-c", "until nc -z mysql 3306; do echo waiting for mysql; sleep 1; done; python connection_example.py"]