"""
Define and build multiple CNN/LSTM based ECG classifiers (compiled).
"""

from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# ----------------------------------------
# Helper convolution block
# ----------------------------------------
def conv_block(x, filters, kernel=3, pool=True):
    x = layers.Conv1D(filters, kernel, activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    if pool:
        x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)
    return x

# ----------------------------------------
# Simple CNN
# ----------------------------------------
def build_simple_cnn(input_shape, num_classes):
    i = layers.Input(shape=input_shape)
    x = conv_block(i, 32)
    x = conv_block(x, 64)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    o = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(i, o, name="Simple_CNN")
    model.compile(optimizer=Adam(1e-3), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
    return model

# ----------------------------------------
# Advanced CNN
# ----------------------------------------
def build_advanced_cnn(input_shape, num_classes):
    i = layers.Input(shape=input_shape)
    x = conv_block(i, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 256)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    o = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(i, o, name="Advanced_CNN")
    model.compile(optimizer=Adam(1e-3), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
    return model

# ----------------------------------------
# LSTM Model
# ----------------------------------------
def build_lstm_model(input_shape, num_classes):
    i = layers.Input(shape=input_shape)
    x = layers.LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(1e-4))(i)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.4)(x)
    o = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(i, o, name="LSTM_Model")
    model.compile(optimizer=Adam(1e-3), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
    return model

# ----------------------------------------
# Hybrid CNN + LSTM
# ----------------------------------------
def build_hybrid_cnn_lstm(input_shape, num_classes):
    i = layers.Input(shape=input_shape)
    x = conv_block(i, 64)
    x = conv_block(x, 128)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.4)(x)
    o = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(i, o, name="Hybrid_CNN_LSTM")
    model.compile(optimizer=Adam(1e-3), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
    return model

# ----------------------------------------
# Ensemble CNN + LSTM
# ----------------------------------------
def build_ensemble_cnn_lstm(input_shape, num_classes):
    i = layers.Input(shape=input_shape)
    c = conv_block(i, 64)
    c = conv_block(c, 128)
    l = layers.LSTM(64)(i)
    x = layers.Concatenate()([layers.Flatten()(c), l])
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    o = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(i, o, name="Ensemble_CNN_LSTM")
    model.compile(optimizer=Adam(1e-3), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
    return model

# ----------------------------------------
# Return all compiled models
# ----------------------------------------
def build_all_models(input_shape, num_classes):
    return {
        "Simple_CNN": build_simple_cnn(input_shape, num_classes),
        "Advanced_CNN": build_advanced_cnn(input_shape, num_classes),
        "LSTM_Model": build_lstm_model(input_shape, num_classes),
        "Hybrid_CNN_LSTM": build_hybrid_cnn_lstm(input_shape, num_classes),
        "Ensemble_CNN_LSTM": build_ensemble_cnn_lstm(input_shape, num_classes),
    }
