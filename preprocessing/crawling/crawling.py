import subprocess

import configparser
config = configparser.ConfigParser()
config.read('config.ini')

archivo_comandos = config['crawling']['archivo_comandos']

with open(archivo_comandos, "r") as archivo:
    for linea in archivo:
        comando = linea.strip() # Eliminar espacios en blanco y saltos de línea
      
        if comando: # Verificar si la línea no está vacía
            try:
                resultado = subprocess.run(comando, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                continue