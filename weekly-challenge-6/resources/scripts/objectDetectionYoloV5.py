import cv2
import numpy as np

MODEL_FILE = "config/yolov5s.onnx"
CLASS_FILE = "config/object_detection_classes_coco.txt"

# caminho para o video a analisar
VIDEO_FILE= "videos/Lisbon_walk.mp4"

CONFIDENCE_THRESHOLD = 0.4 # Threshold para a confianca nas bounding boxes
NMS_THRESHOLD = 0.4        # Threshold para o algoritmo non maximum supression

WIDTH = 640  # largura da imagem no YOLOV5
HEIGHT = 640 # altura da imagem no YOLOV5

FRAME_SKIP = 1 # de quantas em quantas tramas e' aplicado o detetor de objetos


# ler os nomes das classes
with open(CLASS_FILE, 'r') as f:
    class_names = f.read().split('\n')

# gerar cores aleatoriamente para cada uma das classes
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))


# carregar o modelo, neste caso o YOLO
YoloModel = cv2.dnn.readNet(MODEL_FILE)
YoloModel.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
#YoloModel.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# inicializar um stream de video - neste caso e um ficheiro
videoStream = cv2.VideoCapture(VIDEO_FILE)

frameCount=0

# ciclo de leitura do video
while videoStream.isOpened():
   ret, frame = videoStream.read()

   if not ret:
       print("O video chegou ao fim?")
       break

   # so aplica o algoritmo de FRAME_SKIP em FRAME_SKIP tramas de video
   if frameCount % FRAME_SKIP == 0:
       img = frame
       img_height, img_width, channels = img.shape

       # normalizacao para o YoloV5: valores entre 0 e 1 imagens 640x640
       blob = cv2.dnn.blobFromImage(image=img, scalefactor=1/255.0, size=(WIDTH, HEIGHT), swapRB=True)
       YoloModel.setInput(blob)
       predictions = YoloModel.forward()
       outputs = predictions[0]

       print("ouputs shape = ", outputs.shape)
       bboxes = []
       confidences = []
       classIDs = []

       for i in range(outputs.shape[0]):

               detection=outputs[i,:]

               # obter o valor da confianca da bounding box
               confidence=detection[4]

               if confidence>CONFIDENCE_THRESHOLD:

                    scores = detection[5:]

                    # obter a classe com maior confiança
                    classID = np.argmax(scores)

                    fx = img_width / 640
                    fy = img_height / 640

                    bbox_center_x = detection[0] * fx
                    bbox_center_y = detection[1] * fy
                    bbox_width = detection[2] * fx
                    bbox_height = detection[3] * fy

                    bbox_x = int(bbox_center_x - (bbox_width / 2))
                    bbox_y = int(bbox_center_y - (bbox_height / 2))

                    bboxes.append([bbox_x, bbox_y, int(bbox_width), int(bbox_height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

       # Remover bounding boxes adicionais usando  non maximum suppression
       idxs = cv2.dnn.NMSBoxes(bboxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

       if len(idxs) > 0:
           for i in idxs.flatten():
               # extrair as coordenadas e dimensoes das bounding boxes resultantes
               (bbox_x, bbox_y) = (bboxes[i][0], bboxes[i][1])
               (bbox_w, bbox_h) = (bboxes[i][2], bboxes[i][3])

               # colocar retangulos e texto a marcar os objetos identificados
               class_name = class_names[classIDs[i]]
               color = COLORS[classIDs[i]]
               cv2.rectangle(img,  (bbox_x, bbox_y), (bbox_x+bbox_w, bbox_y+bbox_h) , color, 2)
               text = "{}: {:.4f}".format(class_name, confidences[i])
               cv2.putText(img, text, (bbox_x, bbox_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)


       cv2.imshow('image', img)

   frameCount = frameCount + 1

   if cv2.waitKey(10) == ord('q'):
       break

videoStream.release()
cv2.destroyAllWindows()