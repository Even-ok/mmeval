import datetime
import numpy as np
import os.path as osp
import tempfile
import logging
import itertools
from collections import OrderedDict
from json import dump
from typing import Dict, List, Optional, Sequence, Union, Tuple
import multiprocessing
from terminaltables import AsciiTable
from tqdm.contrib import tzip

from mmeval.core.base_metric import BaseMetric
from mmeval.fileio import get_local_path, load, get
from mmeval.metrics.utils.coco_wrapper import COCOPanoptic



try:
    import panopticapi
    from panopticapi.evaluation import VOID, PQStat, OFFSET
    from panopticapi.utils import id2rgb, rgb2id
except ImportError:
    panopticapi = None
    id2rgb = None
    rgb2id = None
    VOID = None
    PQStat = None
    OFFSET = 256 * 256 * 256
logging.basicConfig(level=logging.INFO)


CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    ' truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
    'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff',
    'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light',
    'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield',
    'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow',
    'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile',
    'wall-wood', 'water-other', 'window-blind', 'window-other',
    'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
    'cabinet-merged', 'table-merged', 'floor-other-merged',
    'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged',
    'paper-merged', 'food-other-merged', 'building-other-merged',
    'rock-merged', 'wall-other-merged', 'rug-merged'
]
THING_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
STUFF_CLASSES = [
    'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain',
    'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house',
    'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield',
    'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow',
    'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile',
    'wall-wood', 'water-other', 'window-blind', 'window-other',
    'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
    'cabinet-merged', 'table-merged', 'floor-other-merged',
    'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged',
    'paper-merged', 'food-other-merged', 'building-other-merged',
    'rock-merged', 'wall-other-merged', 'rug-merged'
]

MISSING_ID = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91, 94, 96, 97, 98,
    99, 101, 102, 103, 104, 105, 106, 108, 110, 111, 113, 114, 115, 116, 117,
    120, 121, 123, 124, 126, 127, 129, 131, 132, 134, 135, 136, 137, 139, 140,
    142, 143, 146, 150, 152, 153, 157, 158, 160, 162, 163, 164, 165, 167, 169, 
    170, 172, 173, 174, 179, 182, 183]  


class COCOPanopticMetric(BaseMetric):
    """COCO panoptic segmentation evaluation metric.
    Evaluate PQ, SQ RQ for panoptic segmentation tasks. Please refer to
    https://cocodataset.org/#panoptic-eval for more details.
    Args:
        ann_file (str, optional): Path to the coco format annotation file.
            If not specified, ground truth annotations from the dataset will
            be converted to coco format. Defaults to None.
        seg_prefix (str, optional): Path to the directory which contains the
            coco panoptic segmentation mask. It should be specified when
            evaluate. Defaults to None.
        classwise (bool): Whether to evaluate the metric class-wise.
            Defaults to False.
        outfile_prefix (str, optional): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created.
            It should be specified when format_only is True. Defaults to None.
        nproc (int): Number of processes for panoptic quality computing.
            Defaults to 4. When ``nproc`` exceeds the number of cpu cores,
            the number of cpu cores is used.
        backend_args (dict, optional): Arguments to instantiate the
            preifx of uri corresponding backend. Defaults to None.
        categories (list, optional): The list of catetories of panoramic segmentation.
    """
    def __init__(self,
                 ann_file: Optional[str] = None,
                 seg_prefix: Optional[str] = None,
                 classwise: bool = False,
                 nproc: int = 4,
                 outfile_prefix: Optional[str] = None,
                 backend_args: Optional[dict] = None,
                 categories: list = None,
                 **kwargs) -> None:
        if panopticapi is None:
            raise RuntimeError(
                'panopticapi is not installed, please install it by: '
                'pip install git+https://github.com/cocodataset/'
                'panopticapi.git.')
        super().__init__(**kwargs)
        self.classwise = classwise
        self.tmp_dir = None
        # outfile_prefix should be a prefix of a path which points to a shared
        # storage when train or test with multi nodes.
        self.outfile_prefix = outfile_prefix
        if outfile_prefix is None:
            self.tmp_dir = tempfile.TemporaryDirectory()
            self.outfile_prefix = self.tmp_dir.name 
        
        self.nproc = min(nproc, multiprocessing.cpu_count())
        self.categories = categories  

        if ann_file is not None:
            with get_local_path(
                    filepath=ann_file,
                    backend_args=backend_args) as local_path:
                self._coco_api = COCOPanoptic(local_path)
            self.categories = self._coco_api.cats
        else:
            self._coco_api = None
        
        if self.categories is None:
            # use defalut coco categories
            self.categories = []
            idx = 1
            for name in CLASSES:
                if name in STUFF_CLASSES:
                    isthing = 0
                else:
                    isthing = 1
                while(idx in MISSING_ID): idx += 1
                cat_item = {
                    "supercategory": None,
                    "isthing": isthing,
                    "id": idx,
                    "name" : name
                }
                idx += 1
                self.categories.append(cat_item)


    def convert_to_coco_json(self, ann_list: Sequence[list], img_list: Sequence[list],
                        outfile_prefix: str) -> str:
        """Convert the annotation in list format to coco panoptic segmentation format json file.
        Args:
            ann_list (Sequence[list]): The annotation list.
            img_list (Sequence[list]): The image array list.
            outfile_prefix (str): The filename prefix of the json file. If the
                prefix is "somepath/xxx", the json file will be named
                "somepath/xxx.gt.json".
        Returns:
            str: The filename of the json file.
        """
        assert len(ann_list) > 0, 'gt_dicts is empty.'
        converted_json_path = f'{outfile_prefix}.json'

        image_infos = []
        annotations = []
        segments_info = []
        img_count = 0
        idx = 0
        print("Processing annotations to coco json format...")

        #In the ascending order of image_id
        if 'image_id' in ann_list[0]:
            ann_list.sort(key=lambda x:int(x['image_id']))
            
        for img_ann, pan_png in tzip(ann_list, img_list):
            
            # Build 'images' in json file

            for ann_item in img_ann:
                if 'image_id' not in ann_item.keys():  # only segmentd_info
                    img_count += 1
                    image_info = {
                        'id': img_count,
                        'width': pan_png.shape[1],
                        'height': pan_png.shape[0],
                        'file_name': 'fake_img' + str(img_count) + '.jpg'
                    }
                else:

                    image_info = {
                        'id': ann_item['image_id'],
                        'width': pan_png.shape[1],
                        'height': pan_png.shape[0],
                        'file_name': ann_item['file_name']
                    }
                image_infos.append(image_info)
                break  

            # Build 'annotations' in json file
            pan_png = rgb2id(pan_png)
            segments_info = []
            for each_segment_info in img_ann:

                if 'segments_info' not in each_segment_info.keys():  
                    id = each_segment_info['id']
                    label = each_segment_info['category_id']
                    mask = pan_png == id  
                    isthing = self.categories[label]['isthing']
                    iscrowd = each_segment_info['iscrowd']
                    new_segment_info = {
                        'id': id,
                        'category_id': label,
                        'isthing': isthing,
                        'iscrowd': iscrowd,
                        'area': mask.sum()
                    }
                    segments_info.append(new_segment_info)
                else:
                    segments_info = each_segment_info['segments_info']
                    break

            segm_file = image_info['file_name'].replace('jpg', 'png')
            annotation = dict(
                image_id=image_info['id'],
                segments_info=segments_info,
                file_name=segm_file)
            annotations.append(annotation)
            pan_png = id2rgb(pan_png)



        # Build 'info' in json file
        info = dict(
            date_created=str(datetime.datetime.now()),
            description='Coco json file converted by mmdet CocoPanopticMetric.'
        )

        # Build the final json
        coco_json = dict(
            info=info,
            images=image_infos,
            categories=list(self.categories.values()),
            licenses=None,
        )
        if len(annotations) > 0:
            coco_json['annotations'] = annotations

        with open(converted_json_path, 'w+') as f:
            dump(coco_json, f, ensure_ascii=False, default=self.default_dump)
        return converted_json_path

    def add(self, image_pairs: Sequence[Tuple], ann_pairs: Sequence[Tuple]) -> None:  
        """Add images and annotations.
        Args:
            image_pairs (Sequence[Tuple]): A sequence of tuple. The first element is 
                the list of ground truth image array with shape(W, H, 3), the second 
                element is the list of predicted image array.
            ann_pairs (Sequence[Tuple]): A sequence of tuple. The first element is 
            the annotation of ground truth, which can be 'str' or 'list' to indicate 
            its path or list of annotation. The second element is the annotation of 
            prediction.
            
            If load from
                `ann_file`, the first element can be empty. Else, each element
                represents the annotation for an image, at lease with the following
                keys:
                - id (int): Segmentation id.
                - category_id (int): Category id.
                - iscrowd (int): 0 for single object and 1 for multiple objects.
                - bboxes (numpy.ndarray): Shape (K, 4), the ground truth
                  bounding bboxes of this image, in 'xyxy' foramrt.
        """
        # for image_pair, ann_pair in zip(image_pairs, ann_pairs):
        assert isinstance(image_pairs, tuple), 'The prediciton should be ' \
            f'a sequence of tuple, but got a sequence of {type(image_pairs)}.'  # noqa: E501
        assert isinstance(ann_pairs, tuple), 'The label should be ' \
            f'a sequence of tuple, but got a sequence of {type(ann_pairs)}.'   # noqa: E501
        self.gt_img_list = image_pairs[0]
        self.pred_img_list = image_pairs[1]
        self.gt_ann_list = ann_pairs[0]
        self.pred_ann_list = ann_pairs[1]


    def check_categories(self, categories):
        if isinstance(categories, dict):
            assert "categories" in categories, "The key 'categories' must in the dict"
            return categories
        elif isinstance(categories, list):
            if isinstance(categories[0], str): # only name
                new_cat_list = []
                for name in categories:
                    cat_info = {
                        "name": name,
                        "isthing": 1
                    }
                    new_cat_list.append(cat_info)
                categories = new_cat_list
            if 'id' not in categories[0]:
                for i, item in enumerate(categories):
                    item['id'] = i
                #list2dict =  {el['id']: el for el in categories}

            return {el['id']: el for el in categories}

    def __call__(self, *args, **kwargs) -> Dict:
        """Stateless call for a metric compute."""

        # cache states
        cache_results = self._results
        cache_coco_api = self._coco_api


        self._results = []
        self.gt_img_list = []
        self.pred_img_list = []
        self.gt_ann_list = []
        self.pred_img_list = []
        self.add(*args, **kwargs)
        self.categories = self.check_categories(self.categories)
        metric_result = self.compute_metric()

        # recover states from cache
        self._results = cache_results
        self._coco_api = cache_coco_api

        return metric_result

    
    def default_dump(self, obj):
        """Convert numpy classes to JSON serializable objects."""
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    
    def parse_pq_results(self, pq_results: dict) -> dict:
        """Parse the Panoptic Quality results.
        Args:
            pq_results (dict): Panoptic Quality results.
        Returns:
            dict: Panoptic Quality results parsed.
        """
        result = dict()
        result['PQ'] = 100 * pq_results['All']['pq']
        result['SQ'] = 100 * pq_results['All']['sq']
        result['RQ'] = 100 * pq_results['All']['rq']
        result['PQ_th'] = 100 * pq_results['Things']['pq']
        result['SQ_th'] = 100 * pq_results['Things']['sq']
        result['RQ_th'] = 100 * pq_results['Things']['rq']
        result['PQ_st'] = 100 * pq_results['Stuff']['pq']
        result['SQ_st'] = 100 * pq_results['Stuff']['sq']
        result['RQ_st'] = 100 * pq_results['Stuff']['rq']
        return result


    def print_panoptic_table(self, pq_results, classwise_results=None):
        """Print the panoptic evaluation results table.

        Args:
            pq_results(dict): The Panoptic Quality results.
            classwise_results(dict | None): The classwise Panoptic Quality results.
                The keys are class names and the values are metrics.
        """

        headers = ['', 'PQ', 'SQ', 'RQ', 'categories']
        data = [headers]
        for name in ['All', 'Things', 'Stuff']:
            numbers = [
                f'{(pq_results[name][k] * 100):0.3f}' for k in ['pq', 'sq', 'rq']
            ]
            row = [name] + numbers + [pq_results[name]['n']]
            data.append(row)
        table = AsciiTable(data)
        print('Panoptic Evaluation Results:\n' + table.table)

        if classwise_results is not None:
            class_metrics = [(name, ) + tuple(f'{(metrics[k] * 100):0.3f}'
                                            for k in ['pq', 'sq', 'rq'])
                            for name, metrics in classwise_results.items()]
            num_columns = min(8, len(class_metrics) * 4)
            results_flatten = list(itertools.chain(*class_metrics))
            headers = ['category', 'PQ', 'SQ', 'RQ'] * (num_columns // 4)
            results_2d = itertools.zip_longest(
                *[results_flatten[i::num_columns] for i in range(num_columns)])
            data = [headers]
            data += [result for result in results_2d]
            table = AsciiTable(data)
            print(msg='Classwise Panoptic Evaluation Results:\n' + table.table)

    def pq_compute_single_core(self, proc_id,
                           annotation_set,
                           gt_dict,
                           pred_dict,
                           categories,
                           print_log=False):
        """
        The single core function to evaluate the metric of Panoptic
        Segmentation.
        Similar to the function with the same name in `panopticapi`. 
        Args:
            proc_id (int): The id of the mini process.
            gt_dict (str): The path of the ground truth images.
            pred_dict (str): The path of the prediction images.
            categories (str): The categories of the dataset.
            print_log (bool): Whether to print the log. Defaults to False.
        """
        if PQStat is None:
            raise RuntimeError(
                'panopticapi is not installed, please install it by: '
                'pip install git+https://github.com/cocodataset/'
                'panopticapi.git.')
                
        pq_stat = PQStat()

        idx = 0
        for gt_ann, pred_ann in annotation_set:
            if print_log and idx % 100 == 0:
                print('Core: {}, {} from {} images processed'.format(
                    proc_id, idx, len(annotation_set)))
            idx += 1
            # The gt images can be on the local disk or `ceph`, so we use
            # file_client here.
            pan_gt = gt_dict[gt_ann['image_id']]
            pan_gt = rgb2id(pan_gt)

            pan_pred = pred_dict[pred_ann['image_id']]
            pan_pred = rgb2id(pan_pred)

            gt_segms = {el['id']: el for el in gt_ann['segments_info']}
            pred_segms = {el['id']: el for el in pred_ann['segments_info']}

            # predicted segments area calculation + prediction sanity checks
            pred_labels_set = set(el['id'] for el in pred_ann['segments_info'])
            labels, labels_cnt = np.unique(pan_pred, return_counts=True)
            for label, label_cnt in zip(labels, labels_cnt):
                if label not in pred_segms:
                    if label == VOID:
                        continue
                    raise KeyError(
                        'In the image with ID {} segment with ID {} is '
                        'presented in PNG and not presented in JSON.'.format(
                            gt_ann['image_id'], label))
                pred_segms[label]['area'] = label_cnt
                pred_labels_set.remove(label)
                if pred_segms[label]['category_id'] not in categories:
                    raise KeyError(
                        'In the image with ID {} segment with ID {} has '
                        'unknown category_id {}.'.format(
                            gt_ann['image_id'], label,
                            pred_segms[label]['category_id']))
            if len(pred_labels_set) != 0:
                raise KeyError(
                    'In the image with ID {} the following segment IDs {} '
                    'are presented in JSON and not presented in PNG.'.format(
                        gt_ann['image_id'], list(pred_labels_set)))

            # confusion matrix calculation
            pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(
                np.uint64)
            gt_pred_map = {}
            labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
            for label, intersection in zip(labels, labels_cnt):
                gt_id = label // OFFSET
                pred_id = label % OFFSET
                gt_pred_map[(gt_id, pred_id)] = intersection

            # count all matched pairs
            gt_matched = set()
            pred_matched = set()
            for label_tuple, intersection in gt_pred_map.items():
                gt_label, pred_label = label_tuple
                if gt_label not in gt_segms:
                    continue
                if pred_label not in pred_segms:
                    continue
                if gt_segms[gt_label]['iscrowd'] == 1:
                    continue
                if gt_segms[gt_label]['category_id'] != pred_segms[pred_label][
                        'category_id']:
                    continue

                union = pred_segms[pred_label]['area'] + gt_segms[gt_label][
                    'area'] - intersection - gt_pred_map.get((VOID, pred_label), 0)
                iou = intersection / union
                if iou > 0.5:
                    pq_stat[gt_segms[gt_label]['category_id']].tp += 1
                    pq_stat[gt_segms[gt_label]['category_id']].iou += iou
                    gt_matched.add(gt_label)
                    pred_matched.add(pred_label)

            # count false positives
            crowd_labels_dict = {}
            for gt_label, gt_info in gt_segms.items():
                if gt_label in gt_matched:
                    continue
                # crowd segments are ignored
                if gt_info['iscrowd'] == 1:
                    crowd_labels_dict[gt_info['category_id']] = gt_label
                    continue
                pq_stat[gt_info['category_id']].fn += 1

            # count false positives
            for pred_label, pred_info in pred_segms.items():
                if pred_label in pred_matched:
                    continue
                # intersection of the segment with VOID
                intersection = gt_pred_map.get((VOID, pred_label), 0)
                # plus intersection with corresponding CROWD region if it exists
                if pred_info['category_id'] in crowd_labels_dict:
                    intersection += gt_pred_map.get(
                        (crowd_labels_dict[pred_info['category_id']], pred_label),
                        0)
                # predicted segment is ignored if more than half of
                # the segment correspond to VOID and CROWD regions
                if intersection / pred_info['area'] > 0.5:
                    continue
                pq_stat[pred_info['category_id']].fp += 1

        if print_log:
            print('Core: {}, all {} images processed'.format(
                proc_id, len(annotation_set)))
        return pq_stat

    def pq_compute_multi_core(self, matched_annotations_list,
                            gt_dict,
                            pred_dict,
                            categories,
                            nproc):
        """Evaluate the metrics of Panoptic Segmentation with multithreading.
        Same as the function with the same name in `panopticapi`.
        Args:
            matched_annotations_list (list): The matched annotation list. Each
                element is a tuple of annotations of the same image with the
                format (gt_anns, pred_anns).
            gt_dict (dict): The dictionary that matches the 'image_id' with 
                ground truth image array.
            pred_dict (dict): The dictionary that matches the 'image_id' with 
                predicted image array.
            categories (str): The categories of the dataset.
            nproc (int): Number of processes for panoptic quality computing.
                Defaults to 4. When `nproc` exceeds the number of cpu cores,
                the number of cpu cores is used.
        """

        # cpu_num = min(nproc, multiprocessing.cpu_count())

        annotations_split = np.array_split(matched_annotations_list, nproc)
        print('Number of cores: {}, images per core: {}'.format(
            nproc, len(annotations_split[0])))
        workers = multiprocessing.Pool(processes=nproc)
        processes = []
        for proc_id, annotation_set in enumerate(annotations_split):
            p = workers.apply_async(self.pq_compute_single_core,
                                    (proc_id, annotation_set, gt_dict,
                                    pred_dict, categories))
            processes.append(p)

        # Close the process pool, otherwise it will lead to memory
        # leaking problems.
        workers.close()
        workers.join()

        pq_stat = PQStat()
        for p in processes:
            pq_stat += p.get()

        return pq_stat

    def compute_metric(self, show = True) -> Dict[str, float]:
        """Compute the metrics from processed results.
        Args:
            show (bool, optional): Indicates whether the results are 
                displayed in the console.
        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results.
        """
        self.results = []
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        if self.tmp_dir is not None:
            # do evaluation after collect all the results

            if self._coco_api is None:  # 
                # split gt and prediction list
                gts_json = self.convert_to_coco_json(ann_list=self.gt_ann_list, img_list = self.gt_img_list ,outfile_prefix=osp.join(outfile_prefix, 'gts'))
                # convert predictions to coco format and dump to json file
                pred_json = self.convert_to_coco_json(ann_list=self.pred_ann_list, img_list = self.pred_img_list ,outfile_prefix=osp.join(outfile_prefix, 'pred'))
                self._coco_api = COCOPanoptic(gts_json)
            else:
                if not isinstance(self.pred_ann_list, str):  
                    pred_json = self.convert_to_coco_json(ann_list=self.pred_ann_list, img_list = self.pred_img_list ,outfile_prefix=osp.join(outfile_prefix, 'pred'))
                else:
                    pred_json = self.pred_ann_list

            self.img_ids = self._coco_api.get_img_ids()
            self.categories = self._coco_api.cats

            assert len(self.img_ids) == len(self.gt_img_list)
            assert len(self.gt_img_list) == len(self.pred_img_list)

            self.gt_imgs_id_arrlist = {_id: _arr for _id, _arr in zip(self.img_ids, self.gt_img_list)}
            self.pred_imgs_id_arrlist = {_id: _arr for _id, _arr in zip(self.img_ids, self.pred_img_list)}

            imgs = self._coco_api.imgs
            gt_json = self._coco_api.img_ann_map   
            gt_json = [{
                'image_id': k,
                'segments_info': v,
                'file_name': imgs[k]['segm_file']
            } for k, v in gt_json.items()]
            pred_json = load(pred_json)
            pred_json = dict(
                (el['image_id'], el) for el in pred_json['annotations'])

            # match the gt_anns and pred_anns in the same image
            matched_annotations_list = []
            for gt_ann in gt_json:
                img_id = gt_ann['image_id']
                if img_id not in pred_json.keys():
                    raise Exception('no prediction for the image'
                                    ' with id: {}'.format(img_id))
                matched_annotations_list.append((gt_ann, pred_json[img_id]))
            if self.nproc > 1 and (len(self.img_ids) / self.nproc <= 200 ):   # prevent stack overflow
                pq_stat = self.pq_compute_multi_core(matched_annotations_list,
                    gt_dict=self.gt_imgs_id_arrlist,
                    pred_dict=self.pred_imgs_id_arrlist,
                    categories=self.categories,
                    nproc=self.nproc)
            else:
                pq_stats = self.pq_compute_single_core(
                    proc_id=0,
                    annotation_set=matched_annotations_list,   
                    gt_dict=self.gt_imgs_id_arrlist,
                    pred_dict=self.pred_imgs_id_arrlist,
                    categories=self.categories)

                self.results.append(pq_stats)

                pq_stat = PQStat()
                for result in self.results:
                    pq_stat += result

        metrics = [('All', None), ('Things', True), ('Stuff', False)]
        pq_results = {}

        for name, isthing in metrics:
            pq_results[name], classwise_results = pq_stat.pq_average(
                self.categories, isthing=isthing)
            if name == 'All':
                pq_results['classwise'] = classwise_results

        classwise_results = None
        if self.classwise:
            classwise_results = {
                k: v
                for k, v in zip(self.dataset_meta['classes'],
                                pq_results['classwise'].values())
            }

        if show:
            self.print_panoptic_table(pq_results, classwise_results)
        results = self.parse_pq_results(pq_results)

        return results
