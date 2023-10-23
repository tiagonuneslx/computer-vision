import os
import sys
import tensorflow as tf
from keras import layers
from matplotlib import pyplot as plt
from numpy import mean
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#
#  Instructions
#
# This script takes the path of a directory with labeled images of cats and dogs and
# creates a CNN model to classify images of cats/dogs.
#
# Inside the directory taken as input there should be the following:
#  <directory>/train/cat/         - with images of cats used for training
#  <directory>/train/dog/         - with images of dogs used for training
#  <directory>/validation/cat/    - with images of cats used for testing
#  <directory>/validation/dog/    - with images of dogs used for testing
#

# Definir a seed, para os resultados serem consistentes
tf.keras.utils.set_random_seed(7)

path = input("Enter the path to the images directory: ")

train_path = os.path.join(path, "train")
validation_path = os.path.join(path, "validation")

if not os.path.isdir(path) or not os.path.isdir(train_path) or not os.path.isdir(validation_path):
    sys.exit(f"The directory doesn't exist or it doesn't have the expected structure: {path}.\n"
             f"Please take a look at the instructions comments at the top of the script.")

BATCH_SIZE = 50
IMG_HEIGHT = 100
IMG_WIDTH = 100
EPOCHS = 25

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_path,
    color_mode='rgb',
    labels='inferred',
    label_mode='categorical',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    validation_path,
    color_mode='rgb',
    labels='inferred',
    label_mode='categorical',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# labels inferidas a partir dos nomes dos diretorios
labels = train_ds.class_names
n_labels = len(labels)
print(f"Labels: {labels}")

# optimazacoes para manter a imagens em memoria
train_ds = train_ds.cache()
test_ds = test_ds.cache()

model = tf.keras.models.Sequential([
    layers.Rescaling(1. / 255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.RandomRotation(0.05),
    layers.RandomFlip("horizontal"),
    layers.Conv2D(16, 5, padding='same', activation='relu'),
    layers.Dropout(0.1),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.Dropout(0.1),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 2, padding='same', activation='relu'),
    layers.Dropout(0.2),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(n_labels, activation="softmax")
])

model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

model.summary()

history = model.fit(train_ds, epochs=EPOCHS, validation_data=test_ds)

# opter as predicoes e ground thruth num formato mais facil de tratar para mostrar os resultados
# (um vetor de ids das classes)
y_pred_train = model.predict(train_ds)
y_pred_train = tf.argmax(y_pred_train, axis=1)
y_true_train = tf.concat([y for x, y in train_ds], axis=0)
y_true_train = tf.argmax(y_true_train, axis=1)

y_pred_test = model.predict(test_ds)
y_pred_test = tf.argmax(y_pred_test, axis=1)
y_true_test = tf.concat([y for x, y in test_ds], axis=0)
y_true_test = tf.argmax(y_true_test, axis=1)

# gerar graficos e matriz de confusao
cm_train = confusion_matrix(y_true_train, y_pred_train)
cm_test = confusion_matrix(y_true_test, y_pred_test)
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

# matriz de confusao para os dados de treino
disp = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Training Set')

# matriz de confusao para os dados de teste
disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Validation Set')
plt.show()

# Dar print da melhor accuracy para os dados de teste
best_val_accuracy = max(history.history['val_accuracy'])
mean_val_accuracy = mean(history.history['val_accuracy'])
print(f"Melhor taxa de acertos: {(best_val_accuracy * 100):.2f}%")
print(f"Média taxa de acertos: {(mean_val_accuracy * 100):.2f}%")
