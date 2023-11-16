import cv2
import numpy as np

# caminhios para os ficheiros de configuracao
MODEL_FILE = "config/frozen_inference_graph.pb"
CONFIG_FILE = "config/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt"
CLASS_FILE = "config/object_detection_classes_coco.txt"

# caminho para o video a analisar
VIDEO_FILE="videos/Lisbon_walk.mp4"

# valor de limiar miniar para considerar que as predicoes sao de fato objetos
CONFIDENCE_THRESHOLD = 0.4

# ler os nomes das classes
with open(CLASS_FILE, 'r') as f:
    class_names = f.read().split('\n')

# gerar cores aleatoriamente para cada uma das classes
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

# carregar o modelo (neste caso o SSD)
SSDmodel = cv2.dnn.readNet(model=MODEL_FILE, config=CONFIG_FILE, framework="TensorFlow")

#img = cv2.imread("images/street.jpg")
#img_height, img_width, img_channels = image.shape

# inicializar um stream de video - neste caso e um ficheiro
videoStream = cv2.VideoCapture(VIDEO_FILE)

# ciclo de leitura do video
while videoStream.isOpened():

    # obter a proxima frame do video
    ret, frame = videoStream.read()

    if not ret:
        print("O video chegou ao fim?")
        break

    img = frame
    img_height, img_width, channels = img.shape

    # normalizar com blobFromImage - 300x300 serao as dimensoes das imagens enviadas 'a rede
    blob = cv2.dnn.blobFromImage(image=img, size=(300, 300), swapRB=True)
    SSDmodel.setInput(blob)
    output = SSDmodel.forward()

    for detection in output[0, 0, :, :]:

        # oter o indice de confianca na detecao
        confidence = detection[2]

        if confidence > CONFIDENCE_THRESHOLD:

            # obter a classe
            class_id = detection[1]
            class_name = class_names[int(class_id) - 1]
            color = COLORS[int(class_id)]

            # obter as coordenadas e dimensoes das bounding boxes, normalizadas para coordenadas da imagem
            bbox_x = detection[3] * img_width
            bbox_y = detection[4] * img_height
            bbox_width = detection[5] * img_width
            bbox_height = detection[6] * img_height

            # colocar retangulos e texto a marcar os objetos identificados
            cv2.rectangle(img, (int(bbox_x), int(bbox_y)), (int(bbox_width), int(bbox_height)), color, thickness=2)
            cv2.putText(img, class_name, (int(bbox_x), int(bbox_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('output', img)

    # epera que se carrgue na tecla "q" para terminar
    if cv2.waitKey(10) == ord('q'):
        break

# fechar o stream de video
videoStream.release()
cv2.destroyAllWindows()