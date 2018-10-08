import tensorflow as tf


class MnistClassifier(object):
    def __init__(self,
                 hidden_size=512,
                 classes=2,
                 lr=0.001):
        self.hidden_size = hidden_size
        self.classes = classes
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.built = False
        self._layers = []

    def build(self):
        self.dense1 = tf.keras.layers.Dense(
                                units=self.hidden_size,
                                activation=tf.nn.relu)
        self._layers.append(self.dense1)

        self.dense2 = tf.keras.layers.Dense(
                                units=self.classes,
                                activation=None)
        self._layers.append(self.dense2)

        self.built = True

    @property
    def variables(self):
        v = []
        for l in self._layers:
            v += l.variables
        return v

    def call(self, inputs, is_train=True):
        outputs = tf.reshape(inputs, (-1, 28*28)) / 255.0
        outputs = self.dense1(outputs)
        if is_train:
            outputs = tf.nn.dropout(outputs, 0.2)
        outputs = self.dense2(outputs)
        return outputs

    def loss(self, logits, labels):
        labels = tf.reshape(tf.cast(labels, tf.int32), (-1, ))
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(loss)
        return loss

    def optimize(self, loss, clipped_value=1.0):
        grads = self.optimizer.compute_gradients(loss, self.variables)
        clipped_grads = [(tf.clip_by_value(g, -clipped_value, clipped_value), v) for g, v in grads]
        train_op = self.optimizer.apply_gradients(clipped_grads)
        return train_op

    def predict(self, logits):
        _, indices = tf.nn.top_k(logits, 1, sorted=False)
        return indices

    def __call__(self, inputs, **kwargs):
        with tf.variable_scope("mnist_classifier"):
            if not self.built:
                self.build()
            return self.call(inputs, **kwargs)
