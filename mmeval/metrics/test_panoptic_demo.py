import numpy as np
from coco_panoptic import COCOPanopticMetric
import json 
from PIL import Image
import os


fake_dataset_metas = {
        'CLASSES': tuple([str(i) for i in range(2)])
}

if __name__ == '__main__':

    ann_path = "ground_truth_panoptic.json"

    with open("/home/PJLAB/liangyiwen/Even/datasets/coco2017/annotations/panoptic_val2017.json",'r', encoding='UTF-8') as f:
        groundtruth = json.load(f)
    
    with open("/home/PJLAB/liangyiwen/Even/code/mmeval_exp/mask_result_1223.json",'r', encoding='UTF-8') as f:
        prediction = json.load(f)


    categories = groundtruth['categories']  

    coco_pan_metric = COCOPanopticMetric(
        categories=categories,
        nproc=32
    )

    groundtruth_list = groundtruth['annotations']
    
    #groundtruth['annotations'][0]['segments_info']

    #groundtruth_list = [groundtruth_list,]

    prediction_list = prediction['annotations']

    # img_path = '/home/PJLAB/liangyiwen/Even/code/mmeval_exp/000000581781.png'

    gt_folder = '/home/PJLAB/liangyiwen/Even/datasets/coco2017/annotations/panoptic_val2017'

    groundtruth_img_list = []
    prediction_img_list = []

    # Be sure to follow the picture number from smallest to largest order
    img_list = os.listdir(gt_folder)
    img_list.sort(key=lambda x:int(x.split('.')[0]))  

    for img_name in img_list:
        img_path = os.path.join(gt_folder, img_name)
        img_array = np.array(Image.open(img_path), dtype=np.uint32)
        groundtruth_img_list.append(img_array)

    # groundtruth_img_list = [np.array(Image.open(img_path), dtype=np.uint32) ]

        # Be sure to follow the picture number from smallest to largest order
    pred_folder = '/home/PJLAB/liangyiwen/Even/code/mmeval_exp/panoptic_show'
    pre_img_list = os.listdir(gt_folder)
    pre_img_list.sort(key=lambda x:int(x.split('.')[0]))  

    for img_name in pre_img_list:
        img_path = os.path.join(gt_folder, img_name)
        img_array = np.array(Image.open(img_path), dtype=np.uint32)
        prediction_img_list.append(img_array)
    
    # TODO: 让结果可视化出来， 并且改掉上面的folder! 进行PR提交
    coco_pan_metric(image_pairs=([groundtruth_img_list,], [prediction_img_list,]), ann_pairs=(groundtruth_list, prediction_list))  # doctest: +ELLIPSIS  # noqa: E501