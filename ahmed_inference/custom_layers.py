"""Custom Keras layers extracted from Ahmed's notebook (needed to load the models).

Source: sentimentAnalysisTransFusionGCV2.ipynb (cell 41). Inference only.
"""
import tensorflow as tf
import keras
from keras.layers import Layer, Dense, Dropout, Softmax, LayerNormalization


@keras.saving.register_keras_serializable()
class SelfAttention(Layer):
    def __init__(self, num_units, dropout=0.1, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.num_units = num_units
        self.dropout = Dropout(dropout)
        self.dense_query = Dense(num_units, activation='relu')
        self.dense_key = Dense(num_units, activation='relu')
        self.dense_value = Dense(num_units, activation='relu')
        self.dense_output = Dense(num_units, activation='relu')
        self.softmax = Softmax(axis=-1)

    def call(self, inputs):
        Q = self.dense_query(inputs)
        K = self.dense_key(inputs)
        V = self.dense_value(inputs)
        scores = tf.matmul(Q, K, transpose_b=True)
        scores_scaled = scores / tf.math.sqrt(tf.cast(self.num_units, dtype=tf.float32))
        attention_weights = self.softmax(scores_scaled)
        attention_dropped = self.dropout(attention_weights)
        attended = tf.matmul(attention_dropped, V)
        output = self.dense_output(attended)
        output_dropped = self.dropout(output)
        return output_dropped

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_units": self.num_units})
        return cfg


@keras.saving.register_keras_serializable()
class SelfAttentionTransformer(Layer):
    def __init__(self, num_units, **kwargs):
        super(SelfAttentionTransformer, self).__init__(**kwargs)
        self.num_units = num_units
        self.dense_query = Dense(num_units)
        self.dense_key = Dense(num_units)
        self.dense_value = Dense(num_units)
        self.softmax = Softmax(axis=-1)
        self.layer_norm = LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        Q = self.dense_query(inputs)
        K = self.dense_key(inputs)
        V = self.dense_value(inputs)
        scores = tf.matmul(Q, K, transpose_b=True)
        scores_scaled = scores / tf.math.sqrt(tf.cast(self.num_units, dtype=tf.float32))
        attention_weights = self.softmax(scores_scaled)
        attended = tf.matmul(attention_weights, V)
        output = inputs + attended
        output_normalized = self.layer_norm(output)
        return output_normalized

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_units": self.num_units})
        return cfg
