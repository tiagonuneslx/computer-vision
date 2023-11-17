import cv2
import os

MODEL_FILE = "resources/config/frozen_inference_graph.pb"
CONFIG_FILE = "resources/config/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt"
DUCKS_DIR = "resources/ducks"
CONFIDENCE_THRESHOLD = .5
IOU_THRESHOLD = .5


# obter precision e recall
def get_metrics(ground_truth_bboxes, bboxes, confidence_scores):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    total_predictions = len(bboxes)
    total_ground_truths = len(ground_truth_bboxes)

    best_ious = []

    for i, bbox in enumerate(bboxes):
        confidence = confidence_scores[i]

        # obter o melhor IoU
        best_iou = 0.0
        best_iou_i = -1
        for i, ground_truth_bbox in enumerate(ground_truth_bboxes):
            iou = get_iou(bbox, ground_truth_bbox)

            if iou > best_iou:
                # guardar o valor do melhor IoU
                best_iou = iou
                best_iou_i = i

        best_ious.append(best_iou)

        # assignar bbox como tp ou fp
        if best_iou_i != -1 and best_iou >= IOU_THRESHOLD:
            true_positives += 1
            # remover ground truth bounding box
            ground_truth_bboxes.pop(best_iou_i)
        else:
            false_positives += 1

    # calcular precisao e recall
    precision = true_positives / total_predictions if total_predictions != 0 else 0
    recall = true_positives / total_ground_truths if total_ground_truths != 0 else 0
    mean_iou = sum(best_ious) / len(best_ious)

    return precision, recall, mean_iou


# obter IoU entre duas bounding boxes
def get_iou(box1, box2):
    x1_box1, y1_box1, width_box1, height_box1 = box1
    x1_box2, y1_box2, width_box2, height_box2 = box2

    x2_box1, y2_box1 = x1_box1 + width_box1, y1_box1 + height_box1
    x2_box2, y2_box2 = x1_box2 + width_box2, y1_box2 + height_box2

    # calcular a area de intercecao
    x_intersection = max(0, min(x2_box1, x2_box2) - max(x1_box1, x1_box2))
    y_intersection = max(0, min(y2_box1, y2_box2) - max(y1_box1, y1_box2))
    intersection_area = x_intersection * y_intersection

    # calcular a area de cada bounding box
    area_box1 = (x2_box1 - x1_box1) * (y2_box1 - y1_box1)
    area_box2 = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)

    # calcular a area de uniao
    union_area = area_box1 + area_box2 - intersection_area

    # evitar divisao por 0
    if union_area == 0:
        return 0.0

    # calcular IoU
    iou = intersection_area / union_area

    return iou


# carregar o modelo (neste caso o SSD)
SSDmodel = cv2.dnn.readNet(model=MODEL_FILE, config=CONFIG_FILE, framework="TensorFlow")

# obter as imagens e preparar listas
fileNames = os.listdir(DUCKS_DIR)
fileNames.sort()

precisions = []
recalls = []
mean_ious = []

print("\n File name | Precision  |   Recall   |   Avg IoU  ")
print(" --------- | ---------- | ---------- | ---------- ")

for fileName in fileNames:
    filePath = os.path.join(DUCKS_DIR, fileName)

    img = cv2.imread(filePath)
    img_height, img_width, img_channels = img.shape

    # normalizar com blobFromImage - 300x300 serao as dimensoes das imagens enviadas 'a rede
    blob = cv2.dnn.blobFromImage(image=img, size=(300, 300), swapRB=True)
    SSDmodel.setInput(blob)
    output = SSDmodel.forward()

    ground_truth_bboxes = [(156, 81, 448, 432), (479, 86, 417, 392)]
    bboxes = []
    confidence_scores = []

    for detection in output[0, 0, :, :]:

        # oter o indice de confianca na detecao
        confidence = detection[2]
        confidence_scores.append(confidence)

        if confidence > CONFIDENCE_THRESHOLD:
            # obter a classe
            class_id = detection[1]
            class_name = "drake"
            blue = (255, 0, 0)
            red = (0, 0, 255)

            # obter as coordenadas e dimensoes das bounding boxes, normalizadas para coordenadas da imagem
            bbox_x = int(detection[3] * img_width)
            bbox_y = int(detection[4] * img_height)
            bbox_width = int(detection[5] * img_width - bbox_x)
            bbox_height = int(detection[6] * img_height - bbox_y)
            bbox = (bbox_x, bbox_y, bbox_width, bbox_height)
            bboxes.append(bbox)

            # colocar retangulos vermelhos a marcar os bounding boxes das predictions
            cv2.rectangle(img, bbox, red, thickness=2)

    # colocar retangulos azuis a marcar os bounding boxes do ground truth
    for ground_truth_bbox in ground_truth_bboxes:
        cv2.rectangle(img, ground_truth_bbox, blue, thickness=2)

    precision, recall, mean_iou = get_metrics(ground_truth_bboxes, bboxes, confidence_scores)
    precisions.append(precision)
    recalls.append(recall)
    mean_ious.append(mean_iou)

    print(f"{fileName:^10} | {f'{precision*100:.2f}%':^10} | {f'{recall*100:.2f}%':^10} | {f'{mean_iou*100:.2f}%':^10}")

    # show image with bounding boxes
    cv2.imshow('output', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print(f"\n{'Average of Precisions (not AP):':35s} {sum(precisions) / len(precisions) * 100:.2f}%")
print(f"{'Average of Recalls: ':35s} {sum(recalls) / len(recalls):.2f}%")
print(f"{'Average of Mean IoUs: ':35s} {sum(mean_ious) / len(mean_ious):.2f}%")