# Imagen base con Python 3.10 para compatibilidad con tensorflow
FROM python:3.10-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar los archivos de requerimientos
COPY requirements_docker.txt requirements_docker.txt

# Instalar dependencias del sistema necesarias para OpenCV, Mediapipe y Gradio
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements_docker.txt

# Copiar el modelo y el archivo de la aplicación
COPY gesture_recognition_model.h5 gesture_recognition_model.h5
COPY app_gradio.py app_gradio.py

# Exponer el puerto para Gradio
EXPOSE 7860

# Comando para ejecutar la aplicación
CMD ["python", "app_gradio.py"]
