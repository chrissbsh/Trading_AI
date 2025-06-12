import tensorflow as tf
EPS = tf.keras.backend.epsilon()

print("tf version: ", tf.__version__)

# def focal_loss(gamma=2.0, alpha=None, from_logits=False):
#     """
#     Focal Loss pour classification multi-classe.

#     Paramètres
#     ----------
#     gamma : float              # >0, intensité du focus (γ = 2 est classique)
#     alpha : float ou list/tuple  # pondération par classe (facultatif).
#                                  # - scalaire : même α pour toutes les classes
#                                  # - liste/tuple/ndarray : un α par classe
#     from_logits : bool         # True si y_pred sont des logits, False s'ils
#                                # sont déjà des probabilités (softmax appliqué)
#     """

#     # Préparation d'alpha
#     if alpha is not None:
#         alpha = tf.constant(alpha, dtype=tf.float32)
#     else:
#         alpha = 1.0

#     def loss_fn(y_true, y_pred):
#         # y_true shape (batch,) ou (batch, 1)  ───> cast en int32
#         y_true_ = tf.cast(tf.reshape(y_true, [-1]), tf.int32)

#         # Probabilités
#         if from_logits:
#             y_pred_ = tf.nn.softmax(y_pred, axis=-1)
#         else:
#             y_pred_ = tf.clip_by_value(y_pred, EPS, 1 - EPS)

#         # One-hot pour sélectionner p_t
#         num_classes = tf.shape(y_pred_)[-1]
#         y_true_oh   = tf.one_hot(y_true_, depth=num_classes)

#         # p_t : proba prédite de la vraie classe
#         p_t = tf.reduce_sum(y_true_oh * y_pred_, axis=-1)

#         # Alpha factor (si vecteur → indexation par y_true)
#         if isinstance(alpha, tf.Tensor) and alpha.shape.rank == 1:
#             alpha_t = tf.gather(alpha, y_true_)
#         else:  # scalaire
#             alpha_t = alpha

#         # Focal Loss
#         focal_factor = tf.pow(1.0 - p_t, gamma)
#         loss = -alpha_t * focal_factor * tf.math.log(p_t)

#         return tf.reduce_mean(loss)

#     return loss_fn


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
