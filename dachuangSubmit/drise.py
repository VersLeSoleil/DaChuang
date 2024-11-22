import warnings
import numpy as np
import logging
import math
import cv2
from datetime import datetime
from utils import bbox_iou

now = datetime.now()

# from RMQ import BasicRMQClient

warnings.filterwarnings('ignore')

logname = '/logs/log-{}.log'.format(now.strftime("%Y-%m-%d"))

logging.info('=' * 10 + ' LOG FILE FOR IMAGE ' + '=' * 10)


class DRISE(object):
    def __init__(self, image, sess, grid_size, prob, num_samples=500, batch_size=1):
        self.image = image
        self.num_samples = num_samples
        self.sess = sess
        self.grid_size = grid_size
        self.prob = prob
        self.image_size = (image.shape[1], image.shape[2])
        self.batch_size = batch_size

    def generate_mask(self, ):
        """
        Return a mask with shape [H, W]
        :return: mask generated by bilinear interpolation
        """
        image_h, image_w = self.image_size
        grid_h, grid_w = self.grid_size, self.grid_size

        # Create cell for mask
        cell_w, cell_h = math.ceil(image_w / grid_w), math.ceil(image_h / grid_h)
        up_w, up_h = (grid_w + 1) * cell_w, (grid_h + 1) * cell_h

        # Create {0, 1} mask
        mask = (np.random.uniform(0, 1, size=(grid_h, grid_w)) < self.prob).astype(np.float32)
        # Up-size to get value in [0, 1]
        mask = cv2.resize(mask, (up_w, up_h), interpolation=cv2.INTER_LINEAR)

        # Randomly crop the mask
        offset_w = np.random.randint(0, cell_w)
        offset_h = np.random.randint(0, cell_h)
        mask = mask[offset_h:offset_h + image_h, offset_w:offset_w + image_w]
        return mask

    def mask_image(self, image, mask):
        masked = ((image.astype(np.float32) / 255 * np.dstack([mask] * 3)) * 255).astype(np.uint8)
        return masked

    def explain(self, image, img_input, y_p_boxes, y_p_num_detections, detection_boxes, detection_scores,
                num_detections,
                detection_classes):
        num_objs = int(y_p_num_detections[0])
        h, w = self.image_size
        res = np.zeros((num_objs, h, w), dtype=np.float32)  # ---> shape[num_objs, h, w]
        max_score = np.zeros((num_objs,), dtype=np.float32)  # ---> shape[num_objs,]
        for i in range(0, self.num_samples):
            mask = self.generate_mask()
            # masked = mask * image
            masked = self.mask_image(image, mask)
            input_dict = {img_input: masked}
            p_boxes, p_scores, p_num_detections, p_classes = self.sess.run(
                    [detection_boxes, detection_scores, num_detections, detection_classes],
                    feed_dict=input_dict)
            if int(p_num_detections[0]) == 0:
                continue
            for idx in range(num_objs):
                iou = np.array([bbox_iou(p_boxes[0][k], y_p_boxes[0][idx])
                                for k in range(int(p_num_detections[0]))])
                cos_sin = p_scores[0][0:int(p_num_detections[0])]
                score = cos_sin * iou
                max_score[idx] = max(score)
                res[idx] += mask * max_score[idx]

        return res
