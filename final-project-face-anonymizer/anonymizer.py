import logging
import os

import scipy
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

logging.disable(logging.WARNING)
logging.disable(logging.INFO)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Definir a seed, para os resultados serem consistentes
tf.keras.utils.set_random_seed(7)

BATCH_SIZE = 50
IMG_WIDTH = 224
IMG_HEIGHT = 224
EPOCHS = 10
CONFIDENCE_THRESHOLD = .2
IOU_THRESHOLD = .5


def preprocess_image_and_boxes(image_path, boxes):
    # Load and preprocess the image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    originalWidth, originalHeight = image.shape[:2]
    image = tf.image.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]

    # Scale bounding box coordinates
    scale_y, scale_x = IMG_WIDTH / originalWidth, IMG_HEIGHT / originalHeight

    scaled_boxes = [[box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y] for box in boxes]

    # Convert to a ragged tensor
    scaled_boxes = tf.ragged.constant(scaled_boxes, dtype=tf.float32)

    return image, scaled_boxes


def load_and_preprocess_dataset(images_path, annotations_path):
    mat = scipy.io.loadmat(annotations_path)
    event_list = mat['event_list']
    file_list = mat['file_list']
    face_bbx_list = mat['face_bbx_list']

    image_paths = []
    bounding_boxes = []

    for event_idx, event in enumerate(event_list):
        event_name = event[0][0]
        for file, bbx in zip(file_list[event_idx][0], face_bbx_list[event_idx][0]):
            file_name = file[0][0]
            bbx = bbx[0].astype('int').tolist()

            path = os.path.join(images_path, event_name, file_name + '.jpg')
            image_paths.append(path)
            bounding_boxes.append(bbx)

    # Preprocess and create the dataset
    preprocessed_data = [preprocess_image_and_boxes(path, boxes) for path, boxes in zip(image_paths, bounding_boxes)]
    preprocessed_images = [data[0] for data in preprocessed_data]
    preprocessed_boxes = [data[1] for data in preprocessed_data]

    # Convert lists to tensors
    image_tensor = tf.convert_to_tensor(preprocessed_images)
    boxes_tensor = tf.ragged.stack(preprocessed_boxes)

    # Create the dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_tensor, boxes_tensor))
    dataset = dataset.batch(32)  # Adjust the batch size as needed
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


train_data = load_and_preprocess_dataset(
    "resources/faces/WIDER_train/images",
    "resources/faces/wider_face_split/wider_face_train.mat"
)
val_data = load_and_preprocess_dataset(
    "resources/faces/WIDER_val/images",
    "resources/faces/wider_face_split/wider_face_val.mat"
)

mobileNetModel = MobileNetV3Small(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), include_top=False)
# Importante! Assinalar que não se pretende treinar os pesos do modelo importado
mobileNetModel.trainable = False

model = tf.keras.models.Sequential([
    mobileNetModel,
    layers.Conv2D(4, (1, 1), activation='sigmoid'),
    layers.Reshape((-1, 4)),
])


def iou_loss(y_true, y_pred):
    if isinstance(y_true, tf.RaggedTensor):
        y_true = y_true.flat_values

    if isinstance(y_pred, tf.RaggedTensor):
        y_pred = y_pred.flat_values

    y_true_x_min, y_true_y_min = y_true[..., 0], y_true[..., 1]
    y_true_x_max, y_true_y_max = y_true_x_min + y_true[..., 2], y_true_y_min + y_true[..., 3]

    y_pred_x_min, y_pred_y_min = y_pred[..., 0], y_pred[..., 1]
    y_pred_x_max, y_pred_y_max = y_pred_x_min + y_pred[..., 2], y_pred_y_min + y_pred[..., 3]

    # Calculate intersection coordinates
    inter_x_min = tf.maximum(y_true_x_min, y_pred_x_min)
    inter_y_min = tf.maximum(y_true_y_min, y_pred_y_min)
    inter_x_max = tf.minimum(y_true_x_max, y_pred_x_max)
    inter_y_max = tf.minimum(y_true_y_max, y_pred_y_max)

    # Calculate intersection and union areas
    intersection_area = tf.maximum(inter_x_max - inter_x_min, 0) * tf.maximum(inter_y_max - inter_y_min, 0)
    true_area = (y_true_x_max - y_true_x_min) * (y_true_y_max - y_true_y_min)
    pred_area = (y_pred_x_max - y_pred_x_min) * (y_pred_y_max - y_pred_y_min)
    union_area = true_area + pred_area - intersection_area

    # Compute IoU
    iou = intersection_area / (union_area + tf.keras.backend.epsilon())
    return -tf.reduce_mean(tf.math.log(iou + tf.keras.backend.epsilon()))


model.compile(optimizer='adam', loss=iou_loss)
model.fit(train_data, epochs=EPOCHS, validation_data=val_data)

model.evaluate(val_data)
model.save("face_detector_model")

# define the directory for .pb model
pb_model_path = "models"
# define the name of .pb model
pb_model_name = "frozen_graph.pb"
# create directory for further converted model
os.makedirs(pb_model_path, exist_ok=True)
# get model TF graph
tf_model_graph = tf.function(lambda x: model(x))
# get concrete function
tf_model_graph = tf_model_graph.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
# obtain frozen concrete function
frozen_tf_func = convert_variables_to_constants_v2(tf_model_graph)
# get frozen graph
frozen_tf_func.graph.as_graph_def()
# save full tf model
tf.io.write_graph(graph_or_graph_def=frozen_tf_func.graph,
                  logdir=pb_model_path,
                  name=pb_model_name,
                  as_text=False)
