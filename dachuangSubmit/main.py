import argparse
import glob
import os
import warnings
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf
from skimage import io, img_as_ubyte
from tqdm import tqdm
from utils import get_config, get_model, get_info, get_cam, save_image, draw, create_file
from utils import get_parser
from drise import DRISE
from gradcam import GradCAM
from kde import KDE

warnings.filterwarnings('ignore')
start = datetime.now()


def main(args):
    # ①参数设置
    img_rs, output_tensor, last_conv_tensor, grads, num_sample, NMS = None, None, None, None, None, None
    # 获得xAI_config.json文件
    config_xAI = get_config(args.config_path)
    # 获得model/config/model_config.json文件
    config_models = get_config(config_xAI['Model']['file_config'])

    # 获得model/src/frozen_inference_graph.pb ，即模型！！
    sess, img_input, detection_boxes, detection_scores, num_detections, detection_classes = get_model(
        config_models[0]['model_path'])
    # 交集阈值0.5
    threshold = config_xAI['Model']['threshold']
    # ②创建目录
    create_file(args.output_path)
    create_file(args.output_numpy)

    # ④正式运行：Run xAI for each image
    # glob.glob(f'{args.image_path}/*.jpg') 会获取指定路径下所有以.jpg结尾的文件名。
    # for j in tqdm(sorted(...)): 循环遍历排序后的文件名列表，并使用tqdm显示进度条。
    for j in tqdm(sorted(glob.glob(f'{args.image_path}/*.jpg'))):
        # Load image from input folder and extract ground-truth labels from xml file
        # 读取当前文件名对应的图片。
        image = cv2.imread(j)
        # 将图片从BGR格式转换为RGB格式
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 将处理后的图片重新整形为指定形状
        image = img.reshape(1, img.shape[0], img.shape[1], 3)
        # 会提取当前文件名的基本名称（不包括路径和扩展名）
        name_img = os.path.basename(j).split('.')[0]

        # 尝试从XML文件中获取与当前图片相关的真实标签框信息，如果找不到对应的XML文件，则会捕获FileNotFoundError异常并将gr_truth_boxes设为None
        try:
            gr_truth_boxes = get_info(config_xAI['Model']['folder_xml'] + f'{name_img}.xml')
        except FileNotFoundError:
            gr_truth_boxes = None

        # First stage of model: Extract 300 boxes （模型的第一阶段：提取300个boxes）
        if args.method == 'GradCAM':
            # ———— first stage ——————
            # 预处理
            image_dict = {}
            last_conv_tensor = sess.graph.get_tensor_by_name(config_xAI['CAM']['first_stage']['target'] + ':0')
            output_tensor = sess.graph.get_tensor_by_name(config_xAI['CAM']['first_stage']['output'] + ':0')
            grads = tf.gradients(np.sum(output_tensor[0, :, 1:2]), last_conv_tensor)[0]
            copy_image = image
            copy_img = img

            y_p_boxes, y_p_num_detections = sess.run([detection_boxes, num_detections],
                                                     feed_dict={img_input: image})
            boxs = []
            for i in range(int(y_p_num_detections[0])):
                h_img, w_img = image.shape[1:3]
                x1, x2 = int(y_p_boxes[0][i][1] * w_img), int(y_p_boxes[0][i][3] * w_img)
                y1, y2 = int(y_p_boxes[0][i][0] * h_img), int(y_p_boxes[0][i][2] * h_img)
                boxs.append([x1, y1, x2, y2])
            # Run GradCAM for each class
            gradcam = GradCAM(sess, last_conv_tensor, output_tensor)
            mask = gradcam(image, grads, img_input, 'first_stage', y_p_boxes)
            # Save image and heatmap
            image_dict[args.method], _ = get_cam(img, mask, gr_truth_boxes, threshold, boxs)
            np.save(os.path.join(args.output_numpy, f"{args.method}_{name_img}.npy"), mask)
            save_image(image_dict, os.path.basename(j), args.output_path,
                       index='gradcam_first_stage_full_image')

            # ———— second stage ——————
            # Main part
            # Second stage of model: Detect final boxes containing the nodule(s) （模型的第二阶段：检测包含结节的最终boxes）
            image_dict = {}
            boxs = None
            sess, img_input, detection_boxes, detection_scores, num_detections, detection_classes = get_model(
                config_models[0]['model_path'])
            last_conv_tensor = sess.graph.get_tensor_by_name(config_xAI['CAM']['second_stage']['target'] + ':0')
            output_tensor = sess.graph.get_tensor_by_name(config_xAI['CAM']['second_stage']['output'] + ':0')
            NMS = sess.graph.get_tensor_by_name(config_xAI['CAM']['second_stage']['NMS'] + ':0')
            y_p_boxes, y_p_scores, y_p_num_detections = sess.run([detection_boxes,
                                                                  detection_scores,
                                                                  num_detections],
                                                                 feed_dict={img_input: copy_image})

            index = config_xAI['CAM']['second_stage']['index']  # 为0
            assert y_p_scores[0][index] > args.threshold
            NMS_tensor = sess.run(NMS, feed_dict={img_input: copy_image})
            indices = NMS_tensor[index]
            grads = tf.gradients(output_tensor[0][indices][1], last_conv_tensor)[0]
            # Run GradCAM and save results
            gradcam2 = GradCAM(sess, last_conv_tensor, output_tensor)
            mask2, x1, y1, x2, y2 = gradcam2(copy_image,
                                             grads,
                                             img_input,
                                             'second_stage',
                                             y_p_boxes,
                                             indices=indices,
                                             index=index)  # cam mask
            # Save image and heatmap
            # 从原始图像img中裁剪出一个感兴趣区域，裁剪后的图像存储在image_dict字典中的'predict_box'键对应的值中。
            # 裁剪区域的坐标范围为[y1:y2, x1:x2]，其形状为[H, W, C]，其中H表示高度，W表示宽度，C表示通道数。
            image_dict['predict_box'] = copy_img[y1:y2, x1:x2]  # [H, W, C]
            # 调用gen_cam函数生成CAM（Class Activation Map），并将结果存储在image_dict字典中的键为args.method的值中。
            # gen_cam函数的输入参数包括裁剪后的图像img[y1:y2, x1:x2]、掩模mask、真实框gr_truth_boxes和阈值threshold。
            # CAM结果用于突出图像中与特定类别相关的区域。
            image_dict[args.method], _ = get_cam(copy_img[y1:y2, x1:x2], mask2, gr_truth_boxes, threshold)
            # 将处理后的图像保存到指定路径。调用save_image函数，其中第一个参数为包含处理后图像的image_dict字典，
            # 第二个参数为原始图像文件名的基本名称（去除路径部分），第三个参数为输出路径，
            # 第四个参数index=f'gradcam_2th_stage_box{index}'用于指定保存图像的索引名称。
            save_image(image_dict, os.path.basename(j), args.output_path, index=f'gradcam_2th_stage_box{index}')

        elif args.method == 'DRISE':
            # Extract boxes from session
            image_dict = {}
            y_p_boxes, y_p_scores, y_p_num_detections = sess.run([detection_boxes,
                                                                  detection_scores,
                                                                  num_detections],
                                                                 feed_dict={img_input: image})
            # Run DRISE and save results
            drise = DRISE(image=image, sess=sess, grid_size=8, prob=0.4, num_samples=100)
            rs = drise.explain(image,
                               img_input,
                               y_p_boxes,
                               y_p_num_detections,
                               detection_boxes,
                               detection_scores,
                               num_detections,
                               detection_classes)
            boxs = []
            for i in range(int(y_p_num_detections[0])):
                h_img, w_img = image.shape[1:3]
                x1, x2 = int(y_p_boxes[0][i][1] * w_img), int(y_p_boxes[0][i][3] * w_img)
                y1, y2 = int(y_p_boxes[0][i][0] * h_img), int(y_p_boxes[0][i][2] * h_img)
                boxs.append([x1, y1, x2, y2])
            rs[0] -= np.min(rs[0])
            rs[0] /= (np.max(rs[0]) - np.min(rs[0]))
            np.save(os.path.join(args.output_numpy, f"{args.method}_{name_img}.npy"), rs)
            image_dict[args.method], _ = get_cam(img, rs[0], gr_truth_boxes, threshold, boxs)
            save_image(image_dict, os.path.basename(j), args.output_path, index='drise_result')

        elif args.method == 'KDE':
            # Extract boxes from session
            y_p_boxes, y_p_scores, y_p_num_detections = sess.run([detection_boxes,
                                                                  detection_scores,
                                                                  num_detections],
                                                                 feed_dict={img_input: image})
            # Run KDE and save results
            all_box = None
            kde = KDE(sess, image, j, y_p_num_detections, y_p_boxes)
            box, box_predicted = kde.get_box_predicted(img_input)
            kernel, f = kde.get_kde_map(box)
            np.save(os.path.join(args.output_numpy, f"{args.method}_{name_img}.npy"), f.T)
            kde_score = 1 / kde.get_kde_score(kernel, box_predicted)  # Compute KDE score
            print('kde_score:', kde_score)
            for i in range(300):
                all_box = draw(image, boxs=[box[i]])
            kde.show_kde_map(box_predicted, f, save_file=args.output_path)


if __name__ == '__main__':
    arguments = get_parser()
    main(arguments)
    print(f'Total training time: {datetime.now() - start}')
