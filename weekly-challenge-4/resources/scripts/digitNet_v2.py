# Para desativar as mensgens de INFO e WARNINGS do tensorflow
import logging, os
logging.disable(logging.WARNING)
logging.disable(logging.INFO)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from keras import layers

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

##############################################################################################
# Preparacao do dataset
#
# Ajustar consoante caso se queira carregamento do dataset a partir do sistema de ficheiros (LOCAL_DATASET=True)
# ou carregamento a partir das funçõe buitin do tensorflow/keras (LOCAL_DATASET=False)
# Caso se escolha carregamento local, ajustar os diretorios das imagens de treino e teste
LOCAL_DATASET = False
TRAIN_PATH = "../solvedExercises/mnist/training"
TEST_PATH = "../solvedExercises/mnist/test"

LABELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
IMG_HEIGHT = 28
IMG_WIDTH = 28

TRAIN_AND_VAL_SAMPLES = 60000
TEST_SAMPLES = 10000
NUM_CLASSES = len(LABELS)

TRAIN_VAL_SPLIT = 90 # train and validation split

if not LOCAL_DATASET:
    # Carregamento direto do MNIST
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.reshape(x_train, (TRAIN_AND_VAL_SAMPLES, IMG_HEIGHT, IMG_WIDTH, 1))
    x_test = np.reshape(x_test, (TEST_SAMPLES, IMG_HEIGHT, IMG_WIDTH, 1))

    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

else:
    # Carregamento do MNIST a partir de ficheiros - gera estruturas tf.data.Dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_PATH,
        color_mode='grayscale',
        labels='inferred',
        label_mode='categorical',
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=TRAIN_AND_VAL_SAMPLES)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_PATH,
        color_mode='grayscale',
        labels='inferred',
        label_mode='categorical',
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=TEST_SAMPLES)

    # Transforma, de forma simplificada, a estrutura tf.data.Dataset em matrizes numpy
    # Funciona porque so' ha' um unico batch nos tf.data.Dataset's carregados

    for image, labels in train_ds:
        x_train = image.numpy()
        y_train = labels.numpy()

    for image, labels in test_ds:
        x_test = image.numpy()
        y_test = labels.numpy()

#######################################################################

# normalizacao
x_train = x_train / 255.0
x_test = x_test / 255.0

# split treino / validacao (porque o MNIST original nao tem conjunto de validacao)
split = x_train.shape[0] * TRAIN_VAL_SPLIT // 100
x_val = x_train[split:]
y_val = y_train[split:]
x_train = x_train[:split]
y_train = y_train[:split]

# mostrar dimensoes das matrizes
print("Amostras de treino: " + str(x_train.shape))
print("Output de treino: " + str(y_train.shape))
print("Amostras de validacao: " + str(x_val.shape))
print("Output de validacao: " + str(y_val.shape))
print("Amostras de teste: " + str(x_test.shape))
print("Output de teste: " + str(y_test.shape))


###############################################################################
# Definicao, treino e teste do modelo
#
# definicao do modelo
model = tf.keras.models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# sumario
model.summary()

# compilacao do modelo - escolha do algoritmo de otimizacao e funcao de perda
model.compile(optimizer="adam",
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=["accuracy"])

# treino do modelo
nEpochs = 10
batchSize = 32
history = model.fit(x_train, y_train, batch_size=batchSize, epochs=nEpochs, validation_data=(x_val, y_val))

# obter predicoes e ground truth
output_pred = model.predict(x_test)
y_pred = np.argmax(output_pred, axis=1)
y_true = np.argmax(y_test, axis=1)


# calcular acertos no conjunto de teste
misses = np.count_nonzero(y_true-y_pred)
accuracy = (TEST_SAMPLES - misses) / TEST_SAMPLES

print(f"Falhou {misses} de {TEST_SAMPLES} exemplos")
print(f"Taxa de acertos: {(accuracy*100):.2f}%")

# gerar uma matriz de confusao
cm = confusion_matrix(y_true, y_pred)


########################################################################
# Mostrar figuras

# exemplos de imagens onde falhou
missesIdx = np.flatnonzero(y_true-y_pred)
plt.figure(1, figsize=(8, 8))
for i in range(0,16):
    idx = missesIdx[i]
    image = x_test[idx,:,:,:] * 255
    ax = plt.subplot(4, 4, i+1)
    plt.imshow(image,cmap="gray")
    plt.title(LABELS[y_pred[idx]] + " (" + LABELS[y_true[idx]] + ")")
    plt.axis("off")
plt.subplots_adjust(hspace=0.7)

# evolucao da accuracy
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc="upper left")
plt.grid(True, ls='--')

# evolucao da loss
plt.figure(3)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc="upper right")
plt.grid(True, ls='--')

# matriz de confusao
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
disp.plot(cmap=plt.cm.Blues)
plt.show()