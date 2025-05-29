from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
import tensorflow as tf
import numpy as np
from FFTTransformer import FFTTransformer

class CNNModel:
    def __init__(self, input_shape):
        self.model = self.build_model(input_shape)
    
    def build_model(self, input_shape):
        model = Sequential([
            Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=256, kernel_size=5, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def train(self, X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, class_weight=None):
        return self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, class_weight=class_weight)
    
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
    
    def save_model(self, path="cnn_fft_model.keras"):
        self.model.save(path)
    
    def load_model(self, path="cnn_fft_model.keras"):
        self.model = tf.keras.models.load_model(path)
    
    def predict(self, raw_signals):
        transformer = FFTTransformer()
        transformed_signals = transformer.apply_fft(raw_signals)
        transformed_signals = np.expand_dims(transformed_signals, axis=-1)  # Reshape for CNN input
        return self.model.predict(transformed_signals)
