# 👕 Fashion MNIST Classifier API

Este proyecto implementa un sistema de clasificación de imágenes de ropa utilizando **Deep Learning** y **Transfer Learning** con la arquitectura **Xception**.

El modelo es capaz de clasificar imágenes en 10 categorías diferentes y se despliega mediante una API moderna y rápida utilizando **FastAPI**, contenerizada con **Docker**.

## 📋 Tabla de Contenidos

  - [Características](https://www.google.com/search?q=%23-caracter%C3%ADsticas)
  - [Estructura del Proyecto](https://www.google.com/search?q=%23-estructura-del-proyecto)
  - [Instalación Local](https://www.google.com/search?q=%23-instalaci%C3%B3n-local)
  - [Obtención de Datos y Entrenamiento](https://www.google.com/search?q=%23-obtenci%C3%B3n-de-datos-y-entrenamiento)
  - [Despliegue con Docker](https://www.google.com/search?q=%23-despliegue-con-docker)
  - [Uso de la API](https://www.google.com/search?q=%23-uso-de-la-api)
  - [Clases Soportadas](https://www.google.com/search?q=%23-clases-soportadas)

-----

## 🚀 Características

  * **Modelo:** Red Neuronal Convolucional (CNN) basada en **Xception** pre-entrenada en ImageNet.
  * **Frameworks:** TensorFlow/Keras para el modelado.
  * **API:** FastAPI para inferencia en tiempo real con documentación automática (Swagger UI).
  * **Validación:** Manejo de tipos y validación de archivos automática.
  * **Despliegue:** Docker y Docker Compose para un entorno reproducible.
  * **Configuración:** Gestión de hiperparámetros centralizada en YAML.

-----

## 📂 Estructura del Proyecto

```text
fashion_mnist_project/
│
├── config/
│   └── config.yaml           # Hiperparámetros y rutas
├── data/                     # Datos (Ignorado por git/docker)
├── models/                   # Modelos guardados (.h5)
├── src/
│   ├── api/
│   │   └── main.py           # Servidor FastAPI
│   ├── data/
│   │   └── make_dataset.py   # Script de descarga de datos
│   ├── features/
│   │   └── build_features.py # Preprocesamiento y Augmentation
│   └── models/
│       ├── model_arch.py     # Arquitectura Xception
│       ├── train_model.py    # Script de entrenamiento
│       └── predict.py        # Script de prueba de inferencia
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

-----

## 💻 Instalación Local

Si deseas correr el proyecto directamente en tu máquina (sin Docker):

1.  **Clonar el repositorio:**

    ```bash
    git clone <tu-repo-url>
    cd fashion_mnist_project
    ```

2.  **Crear un entorno virtual (Opcional pero recomendado):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  **Instalar dependencias:**

    ```bash
    pip install -r requirements.txt
    ```

-----

## 📊 Obtención de Datos y Entrenamiento

Antes de levantar la API, necesitas entrenar el modelo (o colocar uno ya entrenado en la carpeta `models/`).

1.  **Descargar el Dataset:**
    El script descargará el dataset "clothing-dataset-small" automáticamente.

    ```bash
    python -m src.data.make_dataset
    ```

2.  **Entrenar el Modelo:**
    Esto usará la configuración de `config/config.yaml`, entrenará la red Xception y guardará el mejor modelo en la carpeta `models/`.

    ```bash
    python -m src.models.train_model
    ```

    *(Nota: Esto puede tardar varios minutos dependiendo de si tienes GPU o CPU).*

-----

## 🐳 Despliegue con Docker

La forma recomendada de ejecutar la API es utilizando Docker Compose. Esto asegura que todas las dependencias sean correctas.

1.  **Construir y levantar el servicio:**

    ```bash
    docker-compose up --build
    ```

2.  **Verificar estado:**
    La API estará disponible en `http://localhost:8000`.

3.  **Detener el servicio:**
    Presiona `Ctrl+C` o ejecuta:

    ```bash
    docker-compose down
    ```

-----

## 🔌 Uso de la API

Una vez que el servidor esté corriendo (localmente o en Docker), puedes interactuar con él.

### Documentación Interactiva (Swagger UI)

Visita **[http://localhost:8000/docs](https://www.google.com/search?q=http://localhost:8000/docs)** en tu navegador para probar los endpoints visualmente.

### Endpoints Principales

#### 1\. Health Check

  * **Método:** `GET`
  * **URL:** `/`
  * **Respuesta:** Mensaje de bienvenida.

#### 2\. Predicción

  * **Método:** `POST`
  * **URL:** `/predict`
  * **Body:** `form-data` con un campo `file` (imagen).

**Ejemplo con `curl`:**

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/ruta/a/tu/imagen/pantalones.jpg;type=image/jpeg'
```

**Respuesta JSON de ejemplo:**

```json
{
  "prediction": "pants",
  "confidence": 0.9854,
  "all_scores": {
    "dress": 0.0001,
    "pants": 0.9854,
    "t-shirt": 0.0021,
    ...
  }
}
```

-----

## 🏷️ Clases Soportadas

El modelo ha sido entrenado para detectar las siguientes 10 categorías de ropa:

1.  `dress` (Vestido)
2.  `hat` (Sombrero)
3.  `longsleeve` (Manga larga)
4.  `outwear` (Ropa de exterior/Abrigo)
5.  `pants` (Pantalones)
6.  `shirt` (Camisa)
7.  `shoes` (Zapatos)
8.  `shorts` (Pantalones cortos)
9.  `skirt` (Falda)
10. `t-shirt` (Camiseta)

-----

### 📚 Referencia

Este proyecto está basado en los conceptos de *Machine Learning Bookcamp* de Alexey Grigorev.