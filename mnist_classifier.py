import tensorflow as tf


class MnistClassifier(object):
    def __init__(self,
                 hidden_size=300,
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
                                activation=tf.nn.relu,
                                use_bias=False,
                                kernel_initializer=tf.initializers.random_normal())
        self._layers.append(self.dense1)

        self.dense2 = tf.keras.layers.Dense(
                                units=self.classes,
                                activation=tf.nn.relu,
                                use_bias=False,
                                kernel_initializer=tf.initializers.random_normal())
        self._layers.append(self.dense2)

        self.built = True

    @property
    def variables(self):
        v = []
        for l in self._layers:
            v += l.variables
        return v

    def call(self, inputs, labels=None):
        outputs = tf.reshape(inputs, (-1, 28*28))
        outputs = self.dense1(outputs)
        outputs = self.dense2(outputs)
        labels = tf.reshape(tf.one_hot(labels, 10), (-1, 10))
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=outputs, labels=labels)
        loss = tf.reduce_mean(loss)
        return loss

    def optimize(self, loss, clipped_value=1.0):
        grads = self.optimizer.compute_gradients(loss, self.variables)
        clipped_grads = [(tf.clip_by_value(g, -clipped_value, clipped_value), v) for g, v in grads]
        train_op = self.optimizer.apply_gradients(clipped_grads)
        return train_op

    def __call__(self, inputs, **kwargs):
        with tf.variable_scope("mnist_classifier"):
            if not self.built:
                self.build()
            return self.call(inputs, **kwargs)