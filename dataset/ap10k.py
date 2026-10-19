from collections import defaultdict
from collections import OrderedDict
import logging
import os

from xtcocotools.coco import COCO
import json_tricks as json
import numpy as np

from .JointsDataset import JointsDataset, Joints256SizeDataset


logger = logging.getLogger(__name__)


class AP10KDataset(JointsDataset):
    '''
    "keypoints": {
        0: 'L_Eye',
        1: 'R_Eye',
        2: 'Nose',
        3: 'Neck',
        4: 'root of tail',
        5: 'L_Shoulder',
        6: 'L_Elbow',
        7: 'L_F_Paw',
        8: 'R_Shoulder',
        9: 'R_Elbow',
        10: 'R_F_Paw,
        11: 'L_Hip',
        12: 'L_Knee',
        13: 'L_B_Paw',
        14: 'R_Hip',
        15: 'R_Knee',
        16: 'R_B_Paw'
    },
	"skeleton": [
        [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
    '''
    def __init__(self, root, image_set, is_train, output_res=32):
        super().__init__(root, image_set, is_train, output_res)
        self.image_width = 512
        self.image_height = 512
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200

        
        self.coco = COCO(os.path.join(self.root, image_set))

        # deal with class names
        cats = [cat['name']
                for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        self.classes_rev = ['background'] + cats
        logger.info('=> classes: {}'.format(self.classes))
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            [
                (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                for cls in self.classes[1:]
            ]
        )

        # load image file names
        self.image_set_index = self._load_image_set_index()
        self.num_images = len(self.image_set_index)
        logger.info('=> num_images: {}'.format(self.num_images))
        self.id2name, self.name2id = self._get_mapping_id_name(
                self.coco.imgs)

        self.num_joints = 17
        self.flip_pairs = [[0, 1], [5, 8], [6, 9], [7, 10], [11, 14], [12, 15], [13, 16]]

        self.joints_weight = np.array(
            [
                1., 1., 1., 1., 1., 1., 1., 1.2, 1.2,
                1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5
            ],
            dtype=np.float32
        ).reshape((self.num_joints, 1))
        self.upper_body_ids = (0, 1, 2, 3, 5, 6, 7, 8, 9)
        self.lower_body_ids = (4, 10, 11, 12, 13, 14, 15, 16)
        self.sigmas = np.array([
            0.025, 0.025, 0.026, 0.035, 0.035, 0.079, 0.072, 0.062, 0.079, 0.072,
            0.062, 0.107, 0.087, 0.089, 0.107, 0.087, 0.089
        ])
        self.db, self.id2Cat = self._get_db()

        logger.info('=> load {} samples'.format(len(self.db)))

    @staticmethod
    def _get_mapping_id_name(imgs):
        """
        Args:
            imgs (dict): dict of image info.

        Returns:
            tuple: Image name & id mapping dicts.

            - id2name (dict): Mapping image id to name.
            - name2id (dict): Mapping image name to id.
        """
        id2name = {}
        name2id = {}
        for image_id, image in imgs.items():
            file_name = image['file_name']
            id2name[image_id] = file_name
            name2id[file_name] = image_id

        return id2name, name2id

    def _get_ann_file_keypoint(self):
        return os.path.join(
            self.root,
            'annotations',
            'ap10k-' + self.image_set + '-split1.json'
        )

    def _load_image_set_index(self):
        """ image id: int """
        image_ids = self.coco.getImgIds()
        return image_ids

    def _get_db(self):
        gt_db, id2Cat = self._load_coco_keypoint_annotations()
        return gt_db, id2Cat

    def _load_coco_keypoint_annotations(self):
        """ ground truth bbox and keypoints """
        gt_db, id2Cat = [], dict()
        for index in self.image_set_index:
            db_tmp, id2Cat_tmp = self._load_coco_keypoint_annotation_kernal(
                index)
            gt_db.extend(db_tmp)
            id2Cat.update({index: id2Cat_tmp})
        return gt_db, id2Cat

    def _load_coco_keypoint_annotation_kernal(self, index):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: db entry
        """
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            if 'bbox' not in obj:
                continue
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if ('area' not in obj or obj['area'] > 0) and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)
        objs = valid_objs

        bbox_id = 0
        rec = []
        id2Cat = []
        for obj in objs:
            if 'keypoints' not in obj:
                continue
            if max(obj['keypoints']) == 0:
                continue
            if 'num_keypoints' in obj and obj['num_keypoints'] == 0:
                continue

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float32)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float32)
            keypoints = np.array(obj['keypoints']).reshape(-1, 3)
            joints_3d[:, :2] = keypoints[:, :2]
            joints_3d_vis[:, :2] = np.minimum(1, keypoints[:, 2:3])

            center, scale = self._box2cs(obj['clean_bbox'][:4])

            image_file = os.path.join(self.root, 'data', self.id2name[index])

            rec.append({
                'image': image_file,
                'center': center,
                'scale': scale,
                'bbox': obj['clean_bbox'][:4],
                'img_id': index,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'bbox_score': 1,
                'bbox_id': bbox_id,
                'category': obj['category_id'],
                'category_name': self.classes_rev[obj['category_id']]
            })

            category = obj['category_id']
            id2Cat.append({
                'image_file': image_file,
                'bbox_id': bbox_id,
                'category': category,
            })
            bbox_id = bbox_id + 1

        return rec, id2Cat

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale