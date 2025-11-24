import yaml
from tensorflow import keras
from src.features.build_features import get_data_generators
from src.models.model_arch import build_xception_model

def train():
    # Cargar configuración
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    # Obtener datos
    train_ds, val_ds = get_data_generators()

    # Construir modelo
    model = build_xception_model(config)

    # Callback para guardar el mejor modelo (Checkpointing) 
    checkpoint = keras.callbacks.ModelCheckpoint(
        "models/xception_v4_large_{epoch:02d}_{val_accuracy:.3f}.h5",
        save_best_only=True,
        monitor="val_accuracy",
        mode="max"
    )

    # Entrenar 
    print("Iniciando entrenamiento...")
    history = model.fit(
        train_ds,
        epochs=config['training']['epochs'],
        validation_data=val_ds,
        callbacks=[checkpoint]
    )

if __name__ == "__main__":
    train()