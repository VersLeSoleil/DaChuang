from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import cv2
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
from skimage import io, img_as_ubyte
from skimage.util import view_as_windows


def placeholder_from_data(numpy_array):
    """
    Creates a placeholder from a numpy array.
    :param numpy_array: a numpy array.
    :return: a tensorflow placeholder.
    """
    if numpy_array is None:
        return None
    return tf.placeholder('float', [None, ] + list(numpy_array.shape[1:]))


def get_info(path):
    """
    Get the ground-truth bounding boxes and labels of the image
    :param path: Path to the xml file
    :return: list of bounding boxes and labels
    """
    gr_truth = []
    root = ET.parse(path).getroot()
    for type_tag in root.findall('object'):
        xmin = int(type_tag.find('bndbox/xmin').text)
        ymin = int(type_tag.find('bndbox/ymin').text)
        xmax = int(type_tag.find('bndbox/xmax').text)
        ymax = int(type_tag.find('bndbox/ymax').text)
        gr_truth.append([xmin, ymin, xmax, ymax])
    return gr_truth


def bbox_iou(boxA, boxB, x1y1x2y2=False):
    """
    Compute the intersection over union by taking the intersection area and dividing it by the sum of prediction +
    ground-truth areas - the interesection area
    :param boxA: array of shape [4*1] = [x1,y1,x2,y2]
    :param boxB: array of shape [4*1] = [x1,y1,x2,y2]
    :param x1y1x2y2: if True, interpret box coordinates as [x1,y1,w,h]
    :return: IoU
    """
    if x1y1x2y2:
        my = min(boxA[0], boxB[0])
        My = max(boxA[2], boxB[2])
        mx = min(boxA[1], boxB[1])
        Mx = max(boxA[3], boxB[3])
        h1 = boxA[2] - boxA[0]
        w1 = boxA[3] - boxA[1]
        h2 = boxB[2] - boxB[0]
        w2 = boxB[3] - boxB[1]
        uw = Mx - mx
        uh = My - my
        cw = w1 + w2 - uw
        ch = h1 + h2 - uh

        if cw <= 0 or ch <= 0:
            return 0.0

        area1 = w1 * h1
        area2 = w2 * h2
        carea = cw * ch
        uarea = area1 + area2 - carea
        return carea / uarea
    else:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou


def get_cam(image, mask, gr_truth_boxes, threshold, boxs=None):
    """
    Generate CAM map
    :param image: [H,W,C],the original image
    :param mask: [H,W], range 0~1
    :param gr_truth_boxes: ground-truth bounding boxes
    :param threshold: threshold to filter the bounding boxes
    :param boxs: [N,4], the bounding boxes
    :return: tuple(cam,heatmap)
    """
    if boxs is None:
        boxs = [[0, 0, 0, 0]]
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    # heatmap = np.float32(heatmap) / 255
    # heatmap = heatmap[..., ::-1]  # gbr to rgb
    image_cam = cv2.addWeighted(heatmap, 0.5, image, 0.5, 0)

    image_cam = draw(image_cam, boxs, threshold, gr_truth_boxes)

    # heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]
    # image_cam = np.float32(image_cam) / 255
    image_cam = image_cam[..., ::-1]
    return image_cam, heatmap


def draw(image, boxs, threshold=None, gr_truth_boxes=None, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on image
    :param image: [H,W,C],the original image
    :param boxs: [N,4], the bounding boxes
    :param threshold: the threshold to filter the bounding boxes
    :param gr_truth_boxes: [N,4], the ground-truth bounding boxes
    :param color: the color of the bounding boxes
    :param thickness: the thickness of the bounding boxes
    :return: image with bounding boxes
    """
    img_draw = image
    if gr_truth_boxes is not None:
        for a in boxs:
            iou = []
            for b in gr_truth_boxes:
                iou.append(bbox_iou(a, b))
                test_iou = any(l > threshold for l in iou)
                if test_iou:
                    color = (0, 0, 255)
                    img_draw = cv2.rectangle(image, (a[0], a[1]), (a[2], a[3]), color, thickness)
                else:
                    color = (255, 0, 0)
                    img_draw = cv2.rectangle(image, (a[0], a[1]), (a[2], a[3]), color, thickness)
    else:
        for b in boxs:
            start_point, end_point = (b[0], b[1]), (b[2], b[3])
            image = cv2.rectangle(image, start_point, end_point, color, thickness)
    return img_draw


def save_image(image_dicts, input_image_name, output_dir, index):
    """
    Save output in folder named results
    :param image_dicts: Dictionary results
    :param input_image_name: Name of original image
    :param output_dir: Path to output directory
    :param index: Index of image
    """
    name_img = os.path.splitext(input_image_name)[0]
    for key, image in image_dicts.items():
        io.imsave(os.path.join(output_dir, f'{name_img}-{key}-{index}.jpg'), img_as_ubyte(image))


def get_config(path_config):
    """
    Get config from json file
    :param path_config: Path to config file
    :return: config
    """
    with open(path_config, 'r') as fin:
        config_xAI = json.load(fin)
    return config_xAI


def get_model(model_path):
    """
    Get model from file
    :param model_path: Path to model file
    :return: model
    """
    # 首先创建了一个新的tf.Graph对象graph，然后使用该图作为默认图
    graph = tf.Graph()
    with graph.as_default():
        # 使用tf.gfile.GFile打开model_path指向的文件，并以二进制模式读取文件内容
        with tf.gfile.GFile(model_path, 'rb') as file:
            # 使用tf.GraphDef()解析文件内容，将其转换为一个图的定义graph_def
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file.read())
            # 使用tf.import_graph_def将graph_def导入到当前的图中
            tf.import_graph_def(graph_def, name='')

            # 然后通过graph.get_tensor_by_name方法获取模型中的各个张量（tensor）
            img_input = graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = graph.get_tensor_by_name('detection_scores:0')
            num_detections = graph.get_tensor_by_name('num_detections:0')
            detection_classes = graph.get_tensor_by_name('detection_classes:0')

            # 最后创建一个tf.Session对象sess，并将该会话与当前的图关联起来
            sess = tf.Session(graph=graph)
        return sess, img_input, detection_boxes, detection_scores, num_detections, detection_classes


def get_tensor_mini(sess, layer_name, image, img_input):
    """
    Get tensor from mini model
    :param sess: Session
    :param layer_name: Name of layer
    :param image: Image
    :param img_input: Input tensor
    :return: tensor units
    """
    print(layer_name)
    layer = sess.graph.get_tensor_by_name(layer_name + ':0')
    units = sess.run(layer, feed_dict={img_input: image})
    return units


def get_center(box):
    center_box_x = np.zeros(len(box))
    center_box_y = np.zeros(len(box))
    for i in range(len(box)):
        center_box_x[i] = int((box[i][2] + box[i][0]) / 2)
        center_box_y[i] = int((box[i][3] + box[i][1]) / 2)
    return center_box_x, center_box_y


def softmax(x):
    f = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return f


def create_file(path):
    """
    Create file/directory if file/directory doesn't exist
    """
    if not os.path.exists(path):
        os.makedirs(path)


def get_parser():
    """
    Parse command line arguments
    :return: parser
    """
    parser = argparse.ArgumentParser(description='xAI for thyroid cancer detection')
    parser.add_argument('--config-path', type=str, default='xAI_config.json')
    parser.add_argument('--method', type=str, default='AdaSISE')
    parser.add_argument('--image-path', type=str, default='data/test_images/')
    parser.add_argument('--threshold', type=float, default=0.6)
    parser.add_argument('--output-path', type=str, default='results/')
    parser.add_argument('--output-numpy', type=str, default='results/')
    return parser.parse_args()
