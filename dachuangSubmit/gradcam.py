# --------------------------------GradCAM---------------------------------------
import cv2
import numpy as np


class GradCAM(object):
    """
    1: GradCAM calculate gradient on two stages
    2: Output tensor: Prediction boxes before Non-Max Suppression (first_stage)
    3: Get index target boxes to backpropagation (second_stage), output: The final prediction of the model
    """

    # 这段代码定义了一个名为GradCAM的类，用于计算和可视化梯度加权类激活图（Gradient-weighted Class Activation Mapping）。
    # 第一个阶段：计算预测框在非极大值抑制之前的梯度。
    # 第二个阶段：获取目标框的索引以进行反向传播，并输出模型的最终预测结果。

    # 类的初始化方法__init__接收三个参数：TensorFlow会话session，卷积层的张量conv_tensor和输出层的张量output_tensor。
    def __init__(self, session, conv_tensor, output_tensor):
        """
        Initialize GradCAM
        :param session: Tensorflow session
        :param conv_tensor: Tensor of convolution layer
        :param output_tensor: Tensor of output layer
        """
        self.sess = session
        self.conv_tensor = conv_tensor
        self.output_tensor = output_tensor

    # 类中定义了__call__方法，用于计算GradCAM
    def __call__(self, imgs, grads, img_input, stage, y_p_boxes, indices=0, index=0):
        """
        Calculate GradCAM
        :param imgs: Input image 输入图像
        :param grads: Gradient of output layer 输出层的梯度
        :param stage: Choose a stage to visualize: first_stage or second_stage 选择要可视化的阶段
        :param indices: Index of target boxes to backpropagation (second_stage) 目标框的索引以进行反向传播（第二阶段）
        :param index: Index of image 图像的索引
        :return: GradCAM explanation 返回目标框的位置信息。
        """
        if stage == 'first_stage':
            # first image in batch
            conv_output, grads_val = self.sess.run([self.conv_tensor, grads], feed_dict={img_input: imgs})
            weights = np.mean(grads_val[indices], axis=(0, 1))
            feature = conv_output[indices]
            cam = feature * weights[np.newaxis, np.newaxis, :]
        else:
            conv_output, grads_val = self.sess.run([self.conv_tensor, grads], feed_dict={img_input: imgs})
            weights = np.mean(grads_val[indices], axis=(0, 1))
            feature = conv_output[indices]
            cam = feature * weights[np.newaxis, np.newaxis, :]

        # cam = np.maximum(cam, 0)
        # cam = np.sum(cam, axis=2)
        # cam -= np.min(cam)
        # cam /= (np.max(cam) - np.min(cam))

        cam = np.sum(cam, axis=2)
        # cam = np.maximum(cam, 0) #Relu
        # Normalize data (0, 1)
        cam -= np.min(cam)
        cam /= (np.max(cam) - np.min(cam))
        h_img, w_img = imgs.shape[1:3]
        x1, x2 = int(y_p_boxes[0][index][1] * w_img), int(y_p_boxes[0][index][3] * w_img)
        y1, y2 = int(y_p_boxes[0][index][0] * h_img), int(y_p_boxes[0][index][2] * h_img)
        # Resize CAM
        if stage == 'first_stage':
            cam = cv2.resize(cam, (w_img, h_img))
            return cam
        else:
            cam = cv2.resize(cam, (x2 - x1, y2 - y1))
            return cam, x1, y1, x2, y2

