import logging, os
logging.disable(logging.WARNING)
logging.disable(logging.INFO)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet import MobileNet

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt


###################################################################
#
# Neste exemplo o dataset e' carregado a partir do sistema de ficheiros
# e apenas e' dividido em treino e validacao

batch_size = 128
img_height = 160
img_width = 160
dataset_path = "flower_photos"

train_ds = tf.keras.utils.image_dataset_from_directory(
  dataset_path,
  labels='inferred',
  label_mode = 'categorical',
  validation_split=0.2,
  subset="training",
  seed=1234,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  dataset_path,
  labels='inferred',
  label_mode = 'categorical',
  validation_split=0.2,
  subset="validation",
  seed=1234,
  image_size=(img_height, img_width),
  batch_size=batch_size)

labels = train_ds.class_names
print(labels)

# mostrar algumas imagens de exemplo
plt.figure(1, figsize=(10, 10))
for x_batch, y_batch in train_ds.take(1):
    for i in range(16):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(x_batch[i].numpy().astype("uint8"))
        plt.title(labels[np.argmax(y_batch[i,:])])
        plt.axis("off")
plt.show()

# otimizacoes de gestao de memoria
train_ds = train_ds.cache()
val_ds = val_ds.cache()
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# importar o MobileNet - considerando apenas a parte de geracao de features dessa rede
MobileNetmodel = MobileNet(input_shape=(160, 160, 3), include_top=False, dropout=0.2)
MobileNetmodel.summary()
MobileNetmodel.trainable = False

# construir o modelo
model = tf.keras.models.Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    MobileNetmodel,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(5, activation="softmax")
])

model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

model.summary()

epochs=5
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)

# opter as predicoes e ground thruth num formato mais facil  de tratar (um vetor de ids das classes)
y_pred = model.predict(val_ds)
y_pred = tf.argmax(y_pred, axis=1)

y_true = tf.concat([y for x, y in val_ds], axis=0)
y_true = tf.argmax(y_true, axis=1)


# gerar  graficos e matriz de confusao
cm = confusion_matrix(y_true, y_pred)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(2, figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

# matriz de confusao
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()