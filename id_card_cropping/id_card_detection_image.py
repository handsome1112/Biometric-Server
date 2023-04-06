import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from PIL import Image
import PIL.ImageDraw as ImageDraw

from id_card_cropping.cropped_img.main import cropping

from id_card_cropping.utils import label_map_util
from id_card_cropping.utils import visualization_utils as vis_util

def cropping_id_card(name, direction):

    card_str = ""
    if direction == 0:
        card_str = "front"
    else :
        card_str = "back"
    MODEL_NAME = 'model'
    IMAGE_NAME = 'static/dataset/' + name + '/id_card.png'

    CWD_PATH = os.getcwd()
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
    PATH_TO_LABELS = os.path.join(CWD_PATH,'data','labelmap.pbtxt')
    PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

    NUM_CLASSES = 1

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.compat.v1.Session(graph=detection_graph)


    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    image = cv2.imread(PATH_TO_IMAGE)
    image_expanded = np.expand_dims(image, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    score, array_coord = vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=3,
        min_score_thresh=0.60)

    ymin, xmin, ymax, xmax = array_coord
    print(score)
    if(score < 0.9):
        return False
    else :
        shape = np.shape(image)
        im_width, im_height = shape[1], shape[0]
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
        
        id_card = Image.open('static/dataset/' + name + '/id_card.png')
        draw = ImageDraw.Draw(id_card)
        draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width = 10, fill="red")
    
        id_card.crop((left, top, right, bottom)).save('static/dataset/' + name + '/' + card_str + '.png', quality=95)

        return True
    