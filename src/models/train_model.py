from tensorflow import keras
from tensorflow.keras.applications.xception import Xception

def build_xception_model(config):
    input_shape = tuple(config['model']['input_shape'])
    learning_rate = config['model']['learning_rate']
    drop_rate = config['model']['dropout_rate']
    
    # 1. Cargar modelo base pre-entrenado en ImageNet 
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False # Congelar pesos 

    # 2. Construir la cabecera del modelo 
    inputs = keras.Input(shape=input_shape)
    base = base_model(inputs, training=False)
    vector = keras.layers.GlobalAveragePooling2D()(base)
    
    # Capa interna densa con activación ReLU
    inner = keras.layers.Dense(100, activation='relu')(vector)
    
    # Dropout para regularización
    drop = keras.layers.Dropout(drop_rate)(inner)
    
    # Capa de salida (10 clases)
    outputs = keras.layers.Dense(10)(drop) # Logits por defecto con from_logits=True

    model = keras.Model(inputs, outputs)

    # 3. Compilar con Adam y Learning Rate ajustado 
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    
    return model