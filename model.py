import tensorflow as tf
import tensorflow_hub as hub

base_model = hub.KerasLayer('https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/5', trainable=False)

class ResNetBase(tf.keras.Model):
    def __init__(self, batch):
        super().__init__()
        self.input_layer = tf.keras.layers.InputLayer((299, 299, 3), batch_size=batch)
        self.resnet = base_model
        self.dropout_1 = tf.keras.layers.Dropout(0.2)
        self.dense_layer_1 = tf.keras.layers.Dense(512, tf.nn.relu, kernel_regularizer=tf.keras.regularizers.L2(0.15))
        self.dropout_2 = tf.keras.layers.Dropout(0.1)
        self.dense_layer_2 = tf.keras.layers.Dense(64, tf.nn.relu, kernel_regularizer=tf.keras.regularizers.L2(0.05))
        self.output_layer = tf.keras.layers.Dense(2)
    
    def call(self, inputs, training=False):
        x = self.input_layer(inputs)
        x = self.resnet(x)
        if training:
            x = self.dropout_1(x, training=training)
        x = self.dense_layer_1(x)
        if training:
            x = self.dropout_2(x, training=training)
        x = self.dense_layer_2(x)
        return self.output_layer(x)