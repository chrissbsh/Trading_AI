import tensorflow as tf
EPS = tf.keras.backend.epsilon()

"""
Ce fichier définit deux métriques personnalisées pour TensorFlow/Keras :
- `F1Macro` : calcule le F1-score macro pour un problème de classification multi-classes.
- `BalancedAcc` : calcule la balanced accuracy (rappel moyen par classe), utile en cas de classes déséquilibrées.

Ces métriques sont utilisées pendant l'entraînement et l'évaluation de modèles pour mieux refléter la performance globale, notamment lorsque l’accuracy simple est trompeuse.
"""


class F1Macro(tf.keras.metrics.Metric):
    def __init__(self, num_classes=3, name="f1_macro", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.tp = self.add_weight(shape=(num_classes,), initializer="zeros", dtype=tf.float32)
        self.fp = self.add_weight(shape=(num_classes,), initializer="zeros", dtype=tf.float32)
        self.fn = self.add_weight(shape=(num_classes,), initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)

        cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes, dtype=tf.float32)
        diag = tf.linalg.tensor_diag_part(cm)

        self.tp.assign_add(diag)
        self.fp.assign_add(tf.reduce_sum(cm, axis=0) - diag)
        self.fn.assign_add(tf.reduce_sum(cm, axis=1) - diag)

    def result(self):
        precision = self.tp / (self.tp + self.fp + EPS)
        recall    = self.tp / (self.tp + self.fn + EPS)
        f1        = 2 * precision * recall / (precision + recall + EPS)
        return tf.reduce_mean(f1)          # macro average

    def reset_states(self):
        for v in (self.tp, self.fp, self.fn):
            v.assign(tf.zeros_like(v))

# ---------- balanced accuracy ----------
class BalancedAcc(tf.keras.metrics.Metric):
    def __init__(self, num_classes=3, name="balanced_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.tp = self.add_weight(shape=(num_classes,), initializer="zeros", dtype=tf.float32)
        self.fn = self.add_weight(shape=(num_classes,), initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)

        cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes, dtype=tf.float32)
        diag = tf.linalg.tensor_diag_part(cm)

        self.tp.assign_add(diag)
        self.fn.assign_add(tf.reduce_sum(cm, axis=1) - diag)

    def result(self):
        recall = self.tp / (self.tp + self.fn + EPS)  # rappel par classe
        return tf.reduce_mean(recall)

    def reset_states(self):
        for v in (self.tp, self.fn):
            v.assign(tf.zeros_like(v))
