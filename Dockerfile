# 1. Imagen base de Python (slim es más ligera)
FROM python:3.9-slim

# 2. Evitar que Python genere archivos .pyc y buffer de salida
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Instalar dependencias del sistema necesarias para TensorFlow y Pillow
# (libgl1-mesa-glx es a menudo necesario para cv2 u operaciones de imagen)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 4. Establecer directorio de trabajo
WORKDIR /app

# 5. Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copiar el código fuente y el modelo
# (Gracias al .dockerignore, ignoramos data/ y notebooks/)
COPY . .

# 7. Exponer el puerto 8000
EXPOSE 8000

# 8. Comando para ejecutar la API
# Usamos la sintaxis de módulo para uvicorn y apuntamos a src.api.main
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]