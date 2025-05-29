import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Dropout, BatchNormalization, 
    LSTM, Bidirectional, Concatenate, GlobalAveragePooling1D, 
    GlobalMaxPooling1D, SpatialDropout1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay

# ✅ Habilitar Precisión Mixta para Mejor Rendimiento
tf.keras.mixed_precision.set_global_policy('mixed_float16')

class UltimateCNNModelGPU:
    def __init__(self, input_shape):
        self.model = self.build_model(input_shape)

    def build_model(self, input_shape):
        input_layer = Input(shape=input_shape)

        # 🔹 **Camino 1: Deep CNN**
        x1 = Conv1D(32, kernel_size=3, padding='same', activation='swish')(input_layer)
        x1 = BatchNormalization()(x1)
        x1 = SpatialDropout1D(0.2)(x1)
        x1 = Conv1D(64, kernel_size=3, padding='same', activation='relu')(x1)
        x1 = Conv1D(128, kernel_size=3, padding='same', activation='relu')(x1)
        x1 = GlobalMaxPooling1D()(x1)

        # 🔹 **Camino 2: BiLSTM con CuDNN (Optimizados)**
        x2 = Bidirectional(LSTM(64, return_sequences=True, implementation=2))(input_layer)  # ✅ CuDNN optimizado
        x2 = Bidirectional(LSTM(64, return_sequences=True, implementation=2))(x2)
        x2 = GlobalAveragePooling1D()(x2)

        # 🔹 **Fusión de Características**
        concatenated = Concatenate()([x1, x2])

        # 🔹 **Capas Densas**
        x = Dense(256, activation='swish')(concatenated)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='swish')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        # 🔹 **Capa de Salida**
        output_layer = Dense(1, activation='sigmoid', dtype='float32')(x)

        # 🔹 **Compilación del Modelo**
        lr_schedule = ExponentialDecay(initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.96)
        optimizer = Adam(learning_rate=lr_schedule)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def train(self, X_train, y_train, epochs=20, batch_size=256, validation_split=0.2, class_weight=None):
        early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

        return self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            class_weight=class_weight,
            callbacks=[early_stopping, reduce_lr]
        )

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def save_model(self, filename):
        self.model.save(filename)
