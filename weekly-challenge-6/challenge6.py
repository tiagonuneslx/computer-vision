import cv2

MODEL_FILE = "resources/config/frozen_inference_graph.pb"
CONFIG_FILE = "resources/config/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt"

# carregar o modelo (neste caso o SSD)
SSDmodel = cv2.dnn.readNet(model=MODEL_FILE, config=CONFIG_FILE, framework="TensorFlow")

# valor de limiar miniar para considerar que as predicoes sao de fato objetos
CONFIDENCE_THRESHOLD = 0.4

img = cv2.imread("resources/ducks/1.jpg")
img_height, img_width, img_channels = img.shape

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
        class_name = "drake"
        color = (255, 0, 0)

        # obter as coordenadas e dimensoes das bounding boxes, normalizadas para coordenadas da imagem
        bbox_x = detection[3] * img_width
        bbox_y = detection[4] * img_height
        bbox_width = detection[5] * img_width
        bbox_height = detection[6] * img_height

        # colocar retangulos e texto a marcar os objetos identificados
        cv2.rectangle(img, (int(bbox_x), int(bbox_y)), (int(bbox_width), int(bbox_height)), color, thickness=2)
        cv2.putText(img, "duck", (int(bbox_x), int(bbox_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

cv2.imshow('output', img)
cv2.waitKey(0)
cv2.destroyAllWindows()