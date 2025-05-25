import tensorflow as tf
print(tf.__version__)
import tensorflow.keras as keras # type: ignore
print(keras.__version__)
print("Num xPUs Available: ", len(tf.config.list_physical_devices('CPU')))
# Si votre NPU est vu comme un autre type de device (ex: TPU), ajustez la chaîne.