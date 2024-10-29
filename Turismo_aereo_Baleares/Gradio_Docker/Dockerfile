# Usa una imagen base de Python
FROM python:3.9-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia el archivo de requisitos y el código de la aplicación a la imagen
COPY requirements.txt requirements.txt
COPY . .

# Instala las dependencias necesarias
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto que Gradio usará
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Comando para iniciar la aplicación
CMD ["python", "app.py"]
