import numpy as np
from coco_detection import COCODetectionMetric
try:
    from utils.coco_wrapper import mask_util
except ImportError as e:
    mask_util = None

num_classes = 4
fake_dataset_metas = {
        'CLASSES': tuple([str(i) for i in range(num_classes)])
    }

coco_det_metric = COCODetectionMetric(
        dataset_meta=fake_dataset_metas,
        metric=['bbox', 'segm']
        )
def _gen_bboxes(num_bboxes, img_w=256, img_h=256):
    # random generate bounding boxes in 'xyxy' formart.
    x = np.random.rand(num_bboxes, ) * img_w
    y = np.random.rand(num_bboxes, ) * img_h
    w = np.random.rand(num_bboxes, ) * (img_w - x)
    h = np.random.rand(num_bboxes, ) * (img_h - y)
    return np.stack([x, y, x + w, y + h], axis=1)

def _gen_masks(bboxes, img_w=256, img_h=256):
    if mask_util is None:
        raise ImportError(
        'Please try to install official pycocotools by '
        '"pip install pycocotools"')
    masks = []
    for i, bbox in enumerate(bboxes):
        mask = np.zeros((img_h, img_w))
        bbox = bbox.astype(np.int32)
        box_mask = (np.random.rand(
            bbox[3] - bbox[1],
            bbox[2] - bbox[0]) > 0.3).astype(np.int)
        mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = box_mask
        masks.append(
                mask_util.encode(
                    np.array(mask[:, :, np.newaxis], order='F',
                                          dtype='uint8'))[0])  # encoded with RLE  为啥要encode呀？
    return masks

if __name__ == '__main__':
    img_id = 1
    img_w, img_h = 256, 256
    num_bboxes = 10
    pred_boxes = _gen_bboxes(
        num_bboxes=num_bboxes,
        img_w=img_w,
        img_h=img_h)
    pred_masks = _gen_masks(
        bboxes=pred_boxes,
        img_w=img_w,
        img_h=img_h)
    prediction = {
        'img_id': img_id,
        'bboxes': pred_boxes,
        'scores': np.random.rand(num_bboxes, ),  # 产生0~1的小数
        'labels': np.random.randint(0, num_classes, size=(num_bboxes, )),
        'masks': pred_masks
        }
    gt_boxes = _gen_bboxes(
        num_bboxes=num_bboxes,
        img_w=img_w,
        img_h=img_h)
    gt_masks = _gen_masks(
        bboxes=pred_boxes,
        img_w=img_w,
        img_h=img_h)
    groundtruth = {
        'img_id': img_id,
        'width': img_w,
        'height': img_h,
        'bboxes': gt_boxes,
        'labels': np.random.randint(0, num_classes, size=(num_bboxes, )),
        'masks': gt_masks,
        'ignore_flags': np.zeros(num_bboxes)
        }
    coco_det_metric(predictions=[prediction, ], groundtruths=[groundtruth, ])  # doctest: +ELLIPSIS  # noqa: E501