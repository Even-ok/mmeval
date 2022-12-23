import os
import os.path as osp
import json
import shutil


if __name__ == '__main__':

    coco_path = '/home/PJLAB/liangyiwen/Even/datasets/coco2017/images'
    new_path = '/home/PJLAB/liangyiwen/Even/datasets/coco2017/panoptic_val_image'
    with open("/home/PJLAB/liangyiwen/Even/datasets/coco2017/annotations/panoptic_val2017.json",'r', encoding='UTF-8') as f:
        groundtruth = json.load(f)

    images_list = groundtruth['images']

    train_list = os.listdir(osp.join(coco_path, "train2017"))
    test_list = os.listdir(osp.join(coco_path, "test2017"))
    val_list = os.listdir(osp.join(coco_path, "val2017"))

    all_list = [train_list,test_list,val_list]
    path_list = [osp.join(coco_path, "train2017"), osp.join(coco_path, "test2017"), osp.join(coco_path, "val2017")]

    for image in images_list:
        file_name = image['file_name']
        for i in range(len(all_list)):
            if file_name in all_list[i]:
                shutil.copy(osp.join(path_list[i], file_name), osp.join(new_path, file_name))