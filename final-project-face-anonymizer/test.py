import cv2
import numpy as np

model = cv2.dnn.readNetFromTensorflow("graph_optimized.pb")

img = cv2.imread("resources/test/img.jpg")

blob = cv2.dnn.blobFromImage(image=img, size=(300, 300), swapRB=True)
model.setInput(blob)
detections = model.forward()

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:  # You can adjust the confidence threshold as needed
        box = detections[0, 0, i, 3:7] * np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        (startX, startY, endX, endY) = box.astype("int")
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # show image with bounding boxes
    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
