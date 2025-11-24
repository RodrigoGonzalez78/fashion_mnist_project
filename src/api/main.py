import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow import keras
from tensorflow.keras.applications.xception import preprocess_input

app = FastAPI(
    title="Fashion MNIST Classifier API",
    description="API para clasificar ropa usando Transfer Learning (Xception)",
    version="1.0"
)

# Variable global para el modelo
model = None

# Clases del dataset (en el orden que el generador las creó)
CLASSES = [
    'dress', 'hat', 'longsleeve', 'outwear', 'pants',
    'shirt', 'shoes', 'shorts', 'skirt', 't-shirt'
]

@app.on_event("startup")
def load_model():
    """Carga el modelo Keras al iniciar el servidor."""
    global model
    # Ruta relativa desde donde se ejecuta el comando uvicorn (raíz del proyecto)
    model_path = "models/xception_v4_large_best.h5" 
    print(f"Cargando modelo desde: {model_path}...")
    try:
        model = keras.models.load_model(model_path)
        print("Modelo cargado exitosamente.")
    except Exception as e:
        print(f"Error cargando el modelo: {e}")

def prepare_image(image_bytes):
    """Convierte bytes en un array numpy preprocesado para Xception."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convertir a RGB si la imagen es escala de grises o tiene canal alfa
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        # Redimensionar a lo que espera el modelo (299x299 según config)
        img = img.resize((299, 299))
        
        # Convertir a array y expandir dimensiones (batch size de 1)
        x = np.array(img)
        x = np.expand_dims(x, axis=0)
        
        # Preprocesamiento específico de Xception
        x = preprocess_input(x)
        return x
    except Exception as e:
        raise HTTPException(status_code=400, detail="Archivo de imagen inválido")

@app.get("/")
def home():
    return {"message": "API de Clasificación de Ropa funcionando. Ve a /docs para probarla."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint para subir una imagen y obtener la predicción.
    """
    # 1. Leer archivo
    contents = await file.read()
    
    # 2. Preprocesar
    if model is None:
        raise HTTPException(status_code=500, detail="El modelo no está cargado")
        
    processed_image = prepare_image(contents)
    
    # 3. Inferencia
    pred = model.predict(processed_image)
    
    # 4. Decodificar resultado
    result_dict = dict(zip(CLASSES, map(float, pred[0])))
    
    # Obtener la clase ganadora
    best_class = CLASSES[pred[0].argmax()]
    confidence = float(pred[0].max())
    
    return {
        "prediction": best_class,
        "confidence": confidence,
        "all_scores": result_dict
    }