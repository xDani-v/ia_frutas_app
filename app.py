from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from keras.models import model_from_json

app = Flask(__name__)

# Cargar la arquitectura del modelo desde el archivo JSON
ruta_archivo_json = "modelo_mobilenetv2.json"
with open(ruta_archivo_json, "r") as archivo_json:
    modelo_cargado_json = archivo_json.read()

# Crear un nuevo modelo a partir de la arquitectura JSON cargada
modelo_cargado = model_from_json(modelo_cargado_json)

# Cargar los pesos en el modelo
modelo_cargado.load_weights("model_mobilenetv2.h5")
IMG_SIZE = (224, 224)
CLASSES = ["banana","manzana","naranja"]

# Función para preprocesar la imagen
def preprocess_image(image):
    image = cv2.resize(image, IMG_SIZE)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Ruta de la página principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para manejar la carga de archivos y realizar predicciones
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No se proporcionó ningún archivo'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'})

    if file:
        # Leer la imagen del archivo
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Preprocesar la imagen
        processed_img = preprocess_image(img)

        # Realizar la predicción
        predictions = modelo_cargado.predict(processed_img)
        predicted_class_index = np.argmax(predictions)
        predicted_class = CLASSES[predicted_class_index]

        return jsonify({'prediction': predicted_class})

# Ruta para activar la cámara y realizar predicciones en cada frame
@app.route('/camera')
def camera():
    return render_template('camera.html')

if __name__ == '__main__':
    app.run(debug=True)
