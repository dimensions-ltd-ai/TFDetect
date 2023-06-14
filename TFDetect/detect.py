import tensorflow as tf
import cv2
import numpy as np
import os.path
import gdown

from .object_detection.utils import label_map_util
import pathlib

parent = pathlib.Path(__file__).parent.resolve()


def load():
    """
    :return: model and category index
    """

    model_path = os.path.join(parent, 'inference_graph_new')
    label_map = os.path.join(parent, 'label_map')
    print('Loading model...', end='')

    if not os.path.exists(model_path):
        print('Downloading model...')
        inference_graph_new = 'https://drive.google.com/drive/folders/1WksUFZAUWI-4ZTUZxSSG4yC64BmOTmbx?usp=sharing'
        gdown.download_folder(inference_graph_new, quiet=False, use_cookies=True, output=str(model_path))

    if not os.path.exists(label_map):
        print('Downloading label map....')
        label_map_path = 'https://drive.google.com/drive/folders/1KEB5lSjNCGM13avUVhx5IA3hZ5cx3JQ_?usp=sharing'
        gdown.download_folder(label_map_path, quiet=False, use_cookies=True, output=str(label_map))

    model_path = os.path.join(model_path, 'saved_model')
    label_map = os.path.join(label_map, 'mscoco_label_map.pbtxt')
    # Load saved model and build the detection function
    model = tf.saved_model.load(model_path)
    print('Done!')

    # Loading the label_map
    category_index = label_map_util.create_category_index_from_labelmap(label_map,
                                                                        use_display_name=True)

    return model, category_index


def detect(frame, model, category_index):
    """

    :param frame: Input Image -> np.array()
    :param model: Tensorflow object detection model ->Tensorflow()
    :param category_index:
    :return:
        :image_rgb: output image with detected boxes -> np.array()
        :final_boxes: detected boundary boxes -> [(x1, y1, x2, y2), ...]
        :classes: name of the detected class -> [class1, class2, ...]
        :scores: predicted confidence of the object class [0-1, 0-1, ...]
    """
    height, width, _ = frame.shape
    tl = round(0.002 * (height + width) / 2) + 1
    ttf = max(tl - 1, 1)
    image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()
    boxes = detections['detection_boxes']
    classes = detections['detection_classes']
    scores = detections['detection_scores']
    final_boxes = []
    for b, c, s in zip(boxes, classes, scores):
        if s > 0.4:
            class_name = category_index[c]['name']
            bbox = tuple(b.tolist())
            ymin, xmin, ymax, xmax = bbox
            box = (int(xmin * width), int(ymin * height), int(xmax * width), int(ymax * height))
            t_size = cv2.getTextSize(class_name, 0, fontScale=tl / 3, thickness=ttf)[0]
            c1 = (box[0], box[1])
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(image_np_with_detections, (box[0], box[1]), (box[2], box[3]), [0, 200, 0], ttf)
            cv2.rectangle(image_np_with_detections, c1, c2, [0, 0, 200], -1, cv2.LINE_AA)  # filled
            cv2.putText(image_np_with_detections, class_name, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255],
                        thickness=ttf, lineType=cv2.LINE_AA)
            final_boxes.append(box)

    image_rgb = cv2.cvtColor(image_np_with_detections, cv2.COLOR_RGB2BGR)
    return image_rgb, final_boxes, classes, scores
