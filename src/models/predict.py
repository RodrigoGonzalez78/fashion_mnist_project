import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import preprocess_input

def predict_image(image_path, model_path):
    # Cargar modelo 
    model = keras.models.load_model(model_path)

    # Cargar y preprocesar imagen a 299x299 
    img = load_img(image_path, target_size=(299, 299))
    x = np.array(img)
    X = np.array([x])
    X = preprocess_input(X)

    # Predicción
    pred = model.predict(X)
    
    # Diccionario de clases (mapeo manual o desde generador) 
    classes = [
        'dress', 'hat', 'longsleeve', 'outwear', 'pants',
        'shirt', 'shoes', 'shorts', 'skirt', 't-shirt'
    ]
    
    # Obtener clase con mayor score 
    result = dict(zip(classes, pred[0]))
    best_class = classes[pred[0].argmax()]
    
    print(f"Predicción: {best_class}")
    return result

if __name__ == "__main__":
    # Ejemplo de uso (ajustar rutas)
    predict_image("data/raw/clothing-dataset-small/test/pants/c8d21106.jpg", 
                  "models/xception_v4_large_best.h5")