# Proyecto de Reconocimiento de Objetos con MobileNetV2

Este proyecto tiene como objetivo desarrollar una aplicación de reconocimiento de objetos utilizando el modelo MobileNetV2. La aplicación carga un modelo preentrenado en MobileNetV2 y permite realizar inferencias en imágenes de la cámara de Windows.

## Estructura del Repositorio

- **/templates**: Carpeta que aloja archivos HTML utilizados por la aplicación.
- **app.py**: Archivo principal que carga el servidor y el modelo de red.
- **main.ipynb**: Jupyter Notebook con el código para utilizar la aplicación con la cámara de Windows.
- **model_mobilenetv2.h5**: Archivo que contiene los pesos del modelo MobileNetV2.
- **modelo_mobilenetv2.json**: Archivo JSON que describe la estructura del modelo MobileNetV2.

## Configuración y Uso

1. Asegúrate de tener todas las dependencias instaladas. Puedes instalarlas utilizando el siguiente comando:

   ```bash
   pip install Flask opencv-python numpy keras
