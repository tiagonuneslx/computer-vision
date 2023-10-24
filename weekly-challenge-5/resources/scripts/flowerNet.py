import logging, os
logging.disable(logging.WARNING)
logging.disable(logging.INFO)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import keras.api._v2.keras as keras
from keras import layers

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

###################################################################
#
# Neste exemplo o dataset e' carregado a partir do sistema de ficheiros
# e apenas e' dividido em treino e validacao

BATCH_SIZE = 32
IMG_HEIGHT = 160
IMG_WIDTH = 160
DATASET_PATH = "../solvedExercises/flower_photos" # ajustar consoante a localizacao
SEED = 1245 # semente do gerador de numeros aleotorios que faz o split treino/validacao
TRAIN_VAL_SPLIT = 0.2 # fracao de imagens para o conjunto de validacao
NUM_CLASSES = 5

train_ds = tf.keras.utils.image_dataset_from_directory(
  DATASET_PATH,
  labels='inferred',
  label_mode = 'categorical',
  validation_split=TRAIN_VAL_SPLIT,
  subset="training",
  seed=SEED,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

val_ds = tf.keras.utils.image_dataset_from_directory(
  DATASET_PATH,
  labels='inferred',
  label_mode = 'categorical',
  validation_split=TRAIN_VAL_SPLIT,
  subset="validation",
  seed=SEED,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

# labels inferidas a partir dos nomes dos diretorios
labels = train_ds.class_names
print(labels)


# plt.figure(1, figsize=(10, 10))
# for x_batch, y_batch in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(x_batch[i].numpy().astype("uint8"))
#         plt.title(labels[np.argmax(y_batch[i,:])])
#         plt.axis("off")
# plt.show()

# optimazacoes para manter a imagens em memoria
train_ds = train_ds.cache()
val_ds = val_ds.cache()

# nota - os layers de data augmentation originam warnings (em versoes do tensorflow superiores a 2.8.3)
# esses warnings sao para ignorar
model = tf.keras.models.Sequential([
    layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.Conv2D(8, 5, padding='same', activation='relu'),
    layers.Dropout(0.2),
    layers.MaxPooling2D(),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.Dropout(0.2),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.Dropout(0.2),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

model.summary()

EPOCHS = 5
history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)

# opter as predicoes e ground thruth num formato mais facil de tratar para mostrar os resultados
# (um vetor de ids das classes)
y_pred = model.predict(val_ds)
y_pred = tf.argmax(y_pred, axis=1)

y_true = tf.concat([y for x, y in val_ds], axis=0)
y_true = tf.argmax(y_true, axis=1)


# gerar graficos e matriz de confusao
cm = confusion_matrix(y_true, y_pred)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(EPOCHS)

# evolucao da loss e acertos
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
