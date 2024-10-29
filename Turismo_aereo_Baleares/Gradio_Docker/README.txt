# Predicción de Turistas por medio aéreo en las Islas Baleares

Este proyecto utiliza **Gradio** y **Prophet** para predecir el número de turistas en Mallorca, Menorca e Ibiza. La predicción se realiza con un modelo de Machine Learning que genera gráficos anuales y mensuales en función de los datos históricos de pasajeros aéreos.

## Requisitos
- Docker instalado en tu máquina.
  - **Descarga Docker Desktop**: Visita [docker.com](https://www.docker.com/products/docker-desktop) y descarga Docker Desktop para tu sistema operativo (Windows, macOS o Linux).
  - **Verifica la instalación**: Ejecuta `docker --version` en la terminal.

## Cómo ejecutar el proyecto

1. Construye la imagen Docker:
  ```bash
  docker build -t gradio-app .

2. Ejecuta de la app:
  ```bash
  docker run -p 7860:7860 gradio-app

3. Accede a la aplicación en tu navegador:
  http://localhost:7860
