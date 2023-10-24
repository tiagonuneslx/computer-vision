import logging, os
logging.disable(logging.WARNING)
logging.disable(logging.INFO)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import keras.api._v2.keras as keras
from keras.applications.vgg16 import VGG16
from keras import utils
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

# obter o modelo VGG16, pre-treinado com o dataset ImageNet
vgg16Model = VGG16(weights='imagenet', classes=1000)

# constantes - 224x224 e' a dimensao das imagens de input na rede VGG16
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_FILE = '../solvedExercises/sampleImages/boat1.jpg'

# obter uma imagem de teste - mudar a gosto - devolve uma imagem em formato PIL
img = utils.load_img(IMG_FILE, target_size=(IMG_HEIGHT, IMG_WIDTH))

# pre-processamento
x = utils.img_to_array(img)       # converte a imagem de PIL para numpy array
x = np.expand_dims(x, axis=0)     # cria uma dimensao adicional para ser compativel com o formato de entrada na CNN
x = preprocess_input(x)           # usa o pre-processamento proprio da VGG16 (converte as imagens para BGR
                                  # e subtrai-lhes a media)

# obter as predicoes
preds = vgg16Model.predict(x)

# processa as predicoes de forma a obter as 5 classes mais provaveis
decoded_preds = decode_predictions(preds, top=5)[0]

# mostra os resultados da classificacao
print('Top-5 Class predictions: ')
for class_name, class_description, score in decoded_preds:
    print(class_description, ": ", score)