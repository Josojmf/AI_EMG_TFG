from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, Dense, Dropout, BatchNormalization,
                                     LSTM, GRU, Bidirectional, Concatenate, GlobalAveragePooling1D,
                                     GlobalMaxPooling1D, SpatialDropout1D)
from tensorflow.keras.activations import swish
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras import mixed_precision
from keras.utils import register_keras_serializable


# Enable Mixed Precision
mixed_precision.set_global_policy('mixed_float16')

@register_keras_serializable()
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads=4, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.num_heads = num_heads

    def build(self, input_shape):
        self.W_q = Dense(input_shape[-1], dtype=tf.float32)
        self.W_k = Dense(input_shape[-1], dtype=tf.float32)
        self.W_v = Dense(input_shape[-1], dtype=tf.float32)
        self.output_dense = Dense(input_shape[-1], dtype=tf.float32)

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        Q = self.W_q(inputs)
        K_ = self.W_k(inputs)
        V = self.W_v(inputs)
        scores = tf.matmul(Q, K_, transpose_b=True) / tf.sqrt(tf.cast(K_.shape[-1], tf.float32))
        attention_weights = tf.nn.softmax(scores, axis=-1)
        output = tf.matmul(attention_weights, V)
        return self.output_dense(output)

class UltimateCNNModel:
    def __init__(self, input_shape):
        self.model = self.build_model(input_shape)

    def build_model(self, input_shape):
        input_layer = Input(shape=input_shape)

        x1 = Conv1D(32, kernel_size=3, padding='same', activation=swish)(input_layer)
        x1 = BatchNormalization()(x1)
        x1 = SpatialDropout1D(0.2)(x1)
        x1 = Conv1D(64, kernel_size=3, padding='same', activation='relu')(x1)
        x1 = Conv1D(128, kernel_size=3, dilation_rate=2, padding='same', activation='relu')(x1)
        x1 = GlobalMaxPooling1D()(x1)

        x2 = Conv1D(32, kernel_size=1, padding='same', activation=swish)(input_layer)
        x2 = BatchNormalization()(x2)
        x2 = Conv1D(64, kernel_size=3, padding='same', activation='relu')(x2)
        x2 = Conv1D(128, kernel_size=3, dilation_rate=2, padding='same', activation='relu')(x2)
        x2 = GlobalAveragePooling1D()(x2)

        x3 = Bidirectional(LSTM(64, return_sequences=True))(input_layer)
        x3 = MultiHeadSelfAttention(num_heads=4)(x3)
        x3 = Bidirectional(GRU(64, return_sequences=True))(x3)
        x3 = GlobalAveragePooling1D()(x3)

        concatenated = Concatenate()([x1, x2, x3])

        x = Dense(256, activation=swish)(concatenated)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        x = Dense(128, activation=swish)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        output_layer = Dense(1, activation='sigmoid', dtype='float32')(x)

        optimizer = Adam(learning_rate=0.001)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def train(self, X_train, y_train, epochs=20, batch_size=256, validation_split=0.2, class_weight=None):
        early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
        return self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            class_weight=class_weight,
            callbacks=[early_stopping]
        )

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def save_model(self, filename):
        self.model.save(filename)
