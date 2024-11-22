import glob
import os
import warnings
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf
from skimage import io, img_as_ubyte
from tqdm import tqdm
from utils import get_config, get_model, get_parser
from xai.drise import DRISE
from xai.gradcam import GradCAM
from xai.kde import KDE

warnings.filterwarnings('ignore')
start = datetime.now()


def energy_based_pointing_game(bbox, saliency_map):
    """
    Calculate energy-based pointing game evaluation
    :param bbox: [N,4], the bounding boxes
    :param saliency_map: [H, W], final saliency map
    """
    h, w = saliency_map.shape
    empty = np.zeros((h, w))
    for b in bbox:
        x1, y1, x2, y2 = b
        # print(x1, y1, x2, y2, h, w)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        h, w = saliency_map.shape
        empty[y1:y2, x1:x2] = 1
    mask_bbox = saliency_map * empty
    energy_bbox =  mask_bbox.sum()
    energy_whole = saliency_map.sum()
    proportion = energy_bbox / energy_whole
    return proportion


def bounding_boxes(bboxs, saliency_map):
    """
    Caculate bounding boxes evaluation
    :param bbox: [N,4], the bounding boxes
    :param saliency_map: [H, W], final saliency map
    """
    height, width = saliency_map.shape
    HW = height*width
    area = 0
    mask = np.zeros((height, width))
    for bbox in bboxs:
        xi, yi, xa, ya = bbox
        area += (xa-xi)*(ya-yi)
        mask[yi:ya, xi:xa] = 1
    sal_order = np.flip(np.argsort(saliency_map.reshape(HW, -1), axis=0), axis=0)
    y= sal_order//saliency_map.shape[1]
    x = sal_order - y*saliency_map.shape[1]
    mask_cam = np.zeros_like(saliency_map)
    mask_cam[y[0:area, :], x[0:area, :]] = 1
    ratio = (mask*mask_cam).sum()/(area)
    return ratio


def IoU(mask, cam_map):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
    area_mask = np.count_nonzero(mask == 1)
    gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # Find contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    mask_cam = np.zeros_like(cam_map)
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        mask_cam[y:y+h, x:x+w] = 1
    area_mask_cam = np.count_nonzero(mask_cam == 1)
    mask_sum = mask*mask_cam
    area_sum = np.count_nonzero(mask_sum)
    iou = area_sum/(area_mask + area_mask_cam - area_sum)
    return iou

def main(args):
    # ---------------------------------Parameters-------------------------------------
    img_rs, output_tensor, last_conv_tensor, grads, num_sample, NMS = None, None, None, None, None, None
    config_xAI = get_config(args.config_path)
    config_models = get_config(config_xAI['Model']['file_config'])
    image_dict = {}
    sess, img_input, detection_boxes, detection_scores, num_detections, detection_classes = get_model(
            config_models[0]['model_path'])
    threshold = config_xAI['Model']['threshold']

    # create array to save results for 5 metrics    
    drop_rate = []
    inc = []
    ebpg_ = []
    bbox_ = []
    iou_ = []

    # Run xAI for each image
    for j in tqdm(sorted(glob.glob(f'{args.image_path}/*.jpg'))):
        # Load image from input folder and extract ground-truth labels from xml file
        image = cv2.imread(j)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = img.reshape(1, img.shape[0], img.shape[1], 3)
        name_img = os.path.basename(j).split('.')[0]
        mu = np.mean(image)
        h_img, w_img = img.shape[:2]
        y_p_boxes, y_p_scores, y_p_num_detections, y_p_classes = sess.run( \
        [detection_boxes, detection_scores, num_detections, detection_classes], \
        feed_dict={img_input: image})
        if y_p_num_detections == 0:
            continue
        # load the saliency map
        cam_map = np.load(os.path.join(args.output_numpy, f"{args.method}_{name_img}.npy"))
        if args.method == 'eLRP':
            cam_map = cam_map[:, :, 2] - cam_map[:, :, 0]
            cam_map = abs(cam_map)
            cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min()) 
        elif args.method == 'D-RISE':
            map = 0
            for cam in cam_map:
                cam = (cam - cam.min()) / (cam.max() - cam.min())
                map += cam
            cam_map = map
        # Coordinates of the predicted boxes
        box_predicted = []
        mask = np.zeros_like(cam_map)
        for i in range(int(y_p_num_detections[0])):
            x1, x2 = int(y_p_boxes[0][i][1] * w_img), int(y_p_boxes[0][i][3] * w_img)
            y1, y2 = int(y_p_boxes[0][i][0] * h_img), int(y_p_boxes[0][i][2] * h_img)
            box_predicted.append([x1,y1,x2,y2])
            mask[y1:y2, x1:x2] = 1  
        # ---------------------------DROP-INC----------------------------
        cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min())
        invert = ((img / 255) * np.dstack([cam_map]*3)) * 255
        bias = mu * (1 -cam_map)
        masked = (invert + bias[:, :, np.newaxis]).astype(np.uint8)
        masked = masked[None, :]
        p_boxes, p_scores, p_num_detections, p_classes = sess.run( \
        [detection_boxes, detection_scores, num_detections, detection_classes], \
        feed_dict={img_input: masked})
        prob = y_p_scores[0][:int(y_p_num_detections[0])].sum()
        prob_ex = p_scores[0][:int(p_num_detections[0])].sum()
        if prob < prob_ex:
            inc.append(1)
        drop = max((prob - prob_ex) / prob, 0)
        drop_rate.append(drop)
        # ---------------------------Localization evaluation----------------------------
        bbox_.append(bounding_boxes(box_predicted, cam_map))
        ebpg_.append(energy_based_pointing_game(box_predicted, cam_map))
        iou_.append(IoU(mask, cam_map))
    # print results with eps = 1e-10 avoid the case where the denominator is 0
    print("Drop rate:", sum(drop_rate)/(len(drop_rate)+1e-10))
    print("Increases", sum(inc)/(len(inc)+1e-10))
    print("EBPG:", sum(ebpg_)/(len(ebpg_)+1e-10))
    print("Bbox:", sum(bbox_)/(len(bbox_)+1e-10))
    print("IoU:", sum(iou_)/(len(iou_)+1e-10))

if __name__ == '__main__':
    arguments = get_parser()
    main(arguments)
    print(f'Total training time: {datetime.now() - start}')    
    