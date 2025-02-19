from tensorflow.keras import layers, Model, optimizers
import tensorflow as tf

def create_model(architecture, input_shape, n_classes):
    inputs = layers.Input(shape=input_shape, dtype=tf.float32)
    
    if architecture == 'BiLSTM':
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(inputs)
        x = layers.Dropout(0.3)(x)
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=False))(x)
    elif architecture == 'BiGRU':
        x = layers.Bidirectional(layers.GRU(256, return_sequences=True))(inputs)
        x = layers.Dropout(0.3)(x)
        x = layers.Bidirectional(layers.GRU(256, return_sequences=True))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Bidirectional(layers.GRU(256, return_sequences=False))(x)
        
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation='softmax', kernel_initializer='he_uniform')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizers.Adam(2e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model