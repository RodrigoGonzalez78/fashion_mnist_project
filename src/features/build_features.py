from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input # [cite: 1683]
import yaml

def get_data_generators(config_path="config/config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    target_size = tuple(config['model']['input_shape'][:2])
    batch_size = config['training']['batch_size']
    data_path = config['data']['raw_path']

    # Augmentation para entrenamiento 
    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        shear_range=10.0,
        zoom_range=0.1,
        horizontal_flip=True
    )

    train_ds = train_gen.flow_from_directory(
        f"{data_path}/train",
        target_size=target_size,
        batch_size=batch_size
    )

    # Validación sin augmentation, solo preprocesamiento 
    val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    val_ds = val_gen.flow_from_directory(
        f"{data_path}/validation",
        target_size=target_size,
        batch_size=batch_size
    )

    return train_ds, val_ds