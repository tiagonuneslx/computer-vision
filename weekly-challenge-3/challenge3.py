import tensorflow as tf
from keras import layers

import matplotlib.pyplot as plt

import numpy as np
from numpy import mean
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

# Definir a seed, para os resultados serem consitentes
tf.keras.utils.set_random_seed(7)

# Utilizar os datasets builtin do tensorflow - facilita a preparacao dos dados
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizacao dos valores de pixel para o intervalo [0 ... 1] - com imagens
# este passo normalmente conduz a resultados melhores
x_train = x_train / 255.0
x_test = x_test / 255.0

# Preparar a ground truth para o formato adequado, usando 10 classes
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Id's das labels e dimensoes das imagens
labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
img_height = 28
img_width = 28

# Mostrar as dimensoes das matrizes para treino e teste
print("Original training set shape:   ", x_train.shape)
print("Original training labels shape:", y_train.shape)

print("Test set shape:                ", x_test.shape)
print("Test labels shape:             ", y_test.shape)

# Visualizar as primeiras 25 imagens do training set original
fig, ax = plt.subplots(5, 5)
for i in range(5):
    for j in range(5):
        ax[i, j].imshow(x_train[i * 5 + j], cmap=plt.get_cmap('gray'))
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)
fig.suptitle('Dataset samples', fontsize=16)
plt.show()

# Desenvolver a partir daqui!

# Criar a CNN
model = tf.keras.models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(560, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(140, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Mostrar um sumario do modelo (organizacao e n. de pesos a otimizar em cada camada)
model.summary()

# compilar o modelo, definindo a loss function e o algoritmo de optimizacao
# legacy.Adam para poder correr eficientemente em Apple M1/M2
model.compile(loss=tf.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
              metrics=['accuracy'])

# treinar, guardando os dados do treino na variavel history
history = model.fit(x_train, y_train, batch_size=56, epochs=30, validation_data=(x_test, y_test))

# obter os id's das classes verdadeiras
y_true = np.argmax(y_test, axis=1)

# realizar as predicoes e obter os id's das classes preditas
output_pred = model(x_test)
y_pred = np.argmax(output_pred, axis=1)

# gerar uma matriz de confusao
cm = confusion_matrix(y_true, y_pred)

# mostrar figuras - accuracy, loss e matriz de confusao
plt.figure(num=1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc="upper left")
plt.grid(True, ls='--')

plt.figure(num=2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc="upper right")
plt.grid(True, ls='--')

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Dar print da melhor accuracy para os dados de teste
best_val_accuracy = max(history.history['val_accuracy'])
mean_val_accuracy = mean(history.history['val_accuracy'])
print(f"Best test accuracy: {best_val_accuracy}")
print(f"Mean test accuracy: {mean_val_accuracy}")
