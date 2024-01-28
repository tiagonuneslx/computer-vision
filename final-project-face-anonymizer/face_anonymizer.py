import cv2
import keras.models
import keras_cv
import numpy as np
import tensorflow as tf
from keras.src.utils.image_utils import ResizeMethod
from matplotlib import pyplot as plt

isTestMode = False

image_files = [
    "resources/test/img1.jpg",
    "resources/test/img2.jpg",
    "resources/test/img3.jpg",
    "resources/test/img4.jpg",
]
class_mapping = {0: 'face'}

model = keras.models.load_model("models/face_detector.keras")


def visualize_detections(model, images, bounding_box_format):
    """
    Use keras_cv utilities to visualize the first few images with the ground truth bounding boxes and the predicted bounding boxes.

    :param model: The TF model
    :param dataset: Dataset to use for detection visualization
    :param bounding_box_format: Bounding box format
    """

    y_pred = model.predict(images)
    y_pred = keras_cv.bounding_box.to_ragged(y_pred)
    keras_cv.visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format=bounding_box_format,
        # y_true=bounding_boxes,
        y_pred=y_pred,
        scale=4,
        rows=2,
        cols=2,
        font_scale=0.7,
        class_mapping=class_mapping,
    )


def load_image(image_path):
    """
    Load JPEG image from path.

    :param image_path: Path to the image file.
    :return: Return image as NumPy array
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.image.resize_with_pad(image,640,640,method=ResizeMethod.BILINEAR)
    image = tf.image.resize(image, (640, 640), method=ResizeMethod.BILINEAR)
    return image


images = []

for image_file in image_files:
    image = load_image(image_file)
    images.append(image)

tf_images = tf.convert_to_tensor(images, dtype=tf.float32)
images = tf_images

if isTestMode:
    visualize_detections(model, images, "xywh")
    plt.show()

y_pred = model.predict(images)
y_pred = keras_cv.bounding_box.to_ragged(y_pred)

red = (0, 0, 255)

for i, image_file in enumerate(image_files):
    image = cv2.imread(image_file)
    img_height, img_width, img_channels = image.shape

    boxes, confidence, classes, num_detections = y_pred["boxes"][i], y_pred["confidence"][i], y_pred["classes"][i], \
        y_pred["num_detections"][i],

    x_adjustment = 0
    y_adjustment = 0
    if img_width < img_height:
        x_adjustment = img_width / img_height
    elif img_width > img_height:
        y_adjustment = img_height / img_width

    for box in boxes:
        w, h = box[2], box[3]

        box = keras_cv.bounding_box.convert_format(
            box,
            images=tf_images[i],
            source="xywh",  # Original Format
            target="xyxy",  # Target Format (to which we want to convert)
        )
        box = box.numpy()

        box[0] = box[0] * (img_width / 640)
        box[1] = box[1] * (img_height / 640)
        box[2] = box[2] * (img_width / 640)
        box[3] = box[3] * (img_height / 640)

        scale_factor = .1  # x1.2

        box[0] = max(0, box[0] - w * scale_factor)
        box[1] = max(0, box[1] - h * scale_factor)
        box[2] = min(img_width, box[2] + w * scale_factor)
        box[3] = min(img_height, box[3] + h * scale_factor)

        start = (box[0], box[1])
        end = (box[2], box[3])
        start = (np.rint(start)).astype(int)
        end = (np.rint(end)).astype(int)
        if isTestMode:
            cv2.rectangle(image, start, end, color=red, thickness=2)

        box = (np.rint(box)).astype(int)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        roi = np.zeros_like(gray)
        roi[box[1]:box[3], box[0]:box[2]] = 255
        mask = cv2.bitwise_and(gray, gray, mask=roi)

        (t, mask) = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU)

        if isTestMode:
            cv2.imshow(image_file, mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Closure
        # Intended to close the gaps inside the coins
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(w * .2), int(h * .2)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, strel)

        if isTestMode:
            cv2.imshow("binary closed", mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Opening
        # Intended to "break the coins away from each other"
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(w * .7), int(h * .7)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, strel)

        if isTestMode:
            cv2.imshow(image_file, mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # mask = 255 - mask

        # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # image = cv2.bitwise_and(image, image, mask=mask)

        blur = cv2.blur(image, (15, 25), 0)
        image[mask > 0] = blur[mask > 0]

    cv2.imshow(image_file, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
