import subprocess

# cambiar directorio de trabajo antes de empezar para que se guarden ahi los datos!!

archivo_comandos = "preprocessing/crawling/BOE.txt"

with open(archivo_comandos, "r") as archivo:
    i = 0
    for linea in archivo:
        comando = linea.strip() # Eliminar espacios en blanco y saltos de línea
      
        if comando: # Verificar si la línea no está vacía
            i = i + 1
            print("Ejecutando comando ", i)
            try:
                resultado = subprocess.run(comando, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                continue

print("Ejecución de ", i, " comandos completada")
