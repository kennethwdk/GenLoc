# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from .transforms import get_affine_transform
from .transforms import affine_transform
from .transforms import fliplr_joints


logger = logging.getLogger(__name__)


class JointsDataset(Dataset):
    def __init__(self, root, image_set, is_train, output_res):
        normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            normalize,
        ])

        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.scale_factor = 0.3
        self.rotation_factor = 40
        self.flip = True
        self.num_joints_half_body = 8
        self.prob_half_body = 0.3
        self.color_rgb = True

        self.image_size = np.array([512, 512])
        self.heatmap_size = np.array([output_res, output_res])
        self.sigma = 2.0
        self.joints_weight = 1

        self.transform = transform
        self.db = []

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''
        img_id = db_rec['img_id']
        if 'category' in db_rec:
            cat_id = db_rec['category']
        else:
            cat_id = 1

        data_numpy = cv2.imread(
            image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )

        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        ori_joints = copy.deepcopy(joints)
        joints_vis = db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1

        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(0.5, 0.5),
        # ])
        # input = transform(input)
        input = torch.from_numpy(input).permute(2, 0, 1).float() / 255

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        heatmap_, mask = self.generate_target(joints, joints_vis)

        heatmap = torch.from_numpy(heatmap_)
        mask = torch.from_numpy(mask)

        stride = self.image_size / self.heatmap_size
        keypoint = torch.from_numpy(joints[:, :2] / stride[0]).float()
        keypoint_mask = torch.from_numpy(joints_vis[:, 0] > 0.0).float()

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'img_id': img_id,
            'cat_id': cat_id,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score,
            'idx': idx,
            'ori_joints': ori_joints,
        }

        return input, heatmap, keypoint, mask, meta

    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        target = np.zeros((self.num_joints,
                            self.heatmap_size[1],
                            self.heatmap_size[0]),
                            dtype=np.float32)

        tmp_size = self.sigma * 3

        for joint_id in range(self.num_joints):
            feat_stride = self.image_size / self.heatmap_size
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight
    
    def _offset_generator(self, joints_3d, joints_vis, num_joints):
        # target_weight = np.ones((num_joints, 1), dtype=np.float32)
        # target_weight[:, 0] = joints_3d[:, 0, 1]
        target = np.zeros((num_joints*2, self.heatmap_size[1], self.heatmap_size[0]),
                          dtype=np.float32)
        target_weight = np.zeros((num_joints*2, self.heatmap_size[1], self.heatmap_size[0]),
                          dtype=np.float32)

        radius = 3.0

        for i in range(num_joints):
            feat_stride = self.image_size / self.heatmap_size
            if joints_vis[i, 0] < 0.5: continue
            x = joints_3d[i, 0] / feat_stride[0]
            y = joints_3d[i, 1] / feat_stride[1]
            if (x >= self.heatmap_size[0] or y >= self.heatmap_size[1] or x < 0 or y < 0): continue

            start_x = max(int(x - radius), 0)
            start_y = max(int(y - radius), 0)
            end_x = min(int(x + radius), self.heatmap_size[0])
            end_y = min(int(y + radius), self.heatmap_size[1])

            for pos_x in range(start_x, end_x):
                for pos_y in range(start_y, end_y):
                    offset_x = x - pos_x
                    offset_y = y - pos_y
                    target[i * 2, pos_y, pos_x] = offset_x
                    target[i * 2 + 1, pos_y, pos_x] = offset_y
                    target_weight[i * 2, pos_y, pos_x] = 1. / radius
                    target_weight[i * 2 + 1, pos_y, pos_x] = 1. / radius

        return target, target_weight

    def _kpt_generator(self, joints_3d, joints_vis, num_joints):
        # target_weight = np.ones((num_joints, 1), dtype=np.float32)
        # target_weight[:, 0] = joints_3d[:, 0, 1]
        target = np.zeros((num_joints*2, self.heatmap_size[1], self.heatmap_size[0]),
                          dtype=np.float32)
        target_weight = np.zeros((num_joints*2, self.heatmap_size[1], self.heatmap_size[0]),
                          dtype=np.float32)

        radius = 3.0

        for i in range(num_joints):
            feat_stride = self.image_size / self.heatmap_size
            if joints_vis[i, 0] < 0.5: continue
            x = joints_3d[i, 0] / feat_stride[0]
            y = joints_3d[i, 1] / feat_stride[1]
            if (x >= self.heatmap_size[0] or y >= self.heatmap_size[1] or x < 0 or y < 0): continue

            start_x = max(int(x - radius), 0)
            start_y = max(int(y - radius), 0)
            end_x = min(int(x + radius), self.heatmap_size[0])
            end_y = min(int(y + radius), self.heatmap_size[1])

            for pos_x in range(start_x, end_x):
                for pos_y in range(start_y, end_y):
                    target[i * 2, pos_y, pos_x] = x  #- 0.5
                    target[i * 2 + 1, pos_y, pos_x] = y  #- 0.5
                    target_weight[i * 2, pos_y, pos_x] = 1. / self.heatmap_size[0]
                    target_weight[i * 2 + 1, pos_y, pos_x] = 1. / self.heatmap_size[1]

        return target, target_weight

    def neighbor_field(self, heatmaps):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        group_idx = [[0, 1, 2, 3, 4], [5, 7, 9], [6, 8, 10], [5, 6], [11, 12], [11, 13, 15], [12, 14, 16]]
        target = np.zeros((self.num_joints,
                            self.heatmap_size[1],
                            self.heatmap_size[0]),
                            dtype=np.float32)

        binary_hm = (heatmaps > 0.3).astype(np.float32)
        for g_idx in group_idx:
            selected_hm = binary_hm[g_idx]
            maxm_hm = np.max(selected_hm, axis=0, keepdims=True)
            maxm_hm = np.repeat(maxm_hm, len(g_idx), axis=0)
            np.maximum(target[g_idx], maxm_hm, out=target[g_idx])

        return target

class Joints256SizeDataset(Dataset):
    def __init__(self, root, image_set, is_train, output_res):
        normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            normalize,
        ])

        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.scale_factor = 0.3
        self.rotation_factor = 40
        self.flip = True
        self.num_joints_half_body = 8
        self.prob_half_body = 0.3
        self.color_rgb = True

        self.image_size = np.array([384, 384])
        self.heatmap_size = np.array([output_res, output_res])
        self.sigma = 2.0
        self.joints_weight = 1

        self.transform = transform
        self.db = []

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''
        img_id = db_rec['img_id']
        if 'category' in db_rec:
            cat_id = db_rec['category']
        else:
            cat_id = 1

        data_numpy = cv2.imread(
            image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )

        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        ori_joints = copy.deepcopy(joints)
        joints_vis = db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1

        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        input = torch.from_numpy(input).permute(2, 0, 1).float() / 255

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        heatmap_, mask = self.generate_target(joints, joints_vis)

        heatmap = torch.from_numpy(heatmap_)
        mask = torch.from_numpy(mask)

        stride = self.image_size / self.heatmap_size
        keypoint = torch.from_numpy(joints[:, :2] / stride[0]).float()
        keypoint_mask = torch.from_numpy(joints_vis[:, 0] > 0.0).float()

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'img_id': img_id,
            'cat_id': cat_id,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score,
            'idx': idx,
            'ori_joints': ori_joints,
        }

        return input, heatmap, keypoint, mask, meta

    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        target = np.zeros((self.num_joints,
                            self.heatmap_size[1],
                            self.heatmap_size[0]),
                            dtype=np.float32)

        tmp_size = self.sigma * 3

        for joint_id in range(self.num_joints):
            feat_stride = self.image_size / self.heatmap_size
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight
    
    def _offset_generator(self, joints_3d, joints_vis, num_joints):
        # target_weight = np.ones((num_joints, 1), dtype=np.float32)
        # target_weight[:, 0] = joints_3d[:, 0, 1]
        target = np.zeros((num_joints*2, self.heatmap_size[1], self.heatmap_size[0]),
                          dtype=np.float32)
        target_weight = np.zeros((num_joints*2, self.heatmap_size[1], self.heatmap_size[0]),
                          dtype=np.float32)

        radius = 3.0

        for i in range(num_joints):
            feat_stride = self.image_size / self.heatmap_size
            if joints_vis[i, 0] < 0.5: continue
            x = joints_3d[i, 0] / feat_stride[0]
            y = joints_3d[i, 1] / feat_stride[1]
            if (x >= self.heatmap_size[0] or y >= self.heatmap_size[1] or x < 0 or y < 0): continue

            start_x = max(int(x - radius), 0)
            start_y = max(int(y - radius), 0)
            end_x = min(int(x + radius), self.heatmap_size[0])
            end_y = min(int(y + radius), self.heatmap_size[1])

            for pos_x in range(start_x, end_x):
                for pos_y in range(start_y, end_y):
                    offset_x = x - pos_x
                    offset_y = y - pos_y
                    target[i * 2, pos_y, pos_x] = offset_x
                    target[i * 2 + 1, pos_y, pos_x] = offset_y
                    target_weight[i * 2, pos_y, pos_x] = 1. / radius
                    target_weight[i * 2 + 1, pos_y, pos_x] = 1. / radius

        return target, target_weight

    def _kpt_generator(self, joints_3d, joints_vis, num_joints):
        # target_weight = np.ones((num_joints, 1), dtype=np.float32)
        # target_weight[:, 0] = joints_3d[:, 0, 1]
        target = np.zeros((num_joints*2, self.heatmap_size[1], self.heatmap_size[0]),
                          dtype=np.float32)
        target_weight = np.zeros((num_joints*2, self.heatmap_size[1], self.heatmap_size[0]),
                          dtype=np.float32)

        radius = 3.0

        for i in range(num_joints):
            feat_stride = self.image_size / self.heatmap_size
            if joints_vis[i, 0] < 0.5: continue
            x = joints_3d[i, 0] / feat_stride[0]
            y = joints_3d[i, 1] / feat_stride[1]
            if (x >= self.heatmap_size[0] or y >= self.heatmap_size[1] or x < 0 or y < 0): continue

            start_x = max(int(x - radius), 0)
            start_y = max(int(y - radius), 0)
            end_x = min(int(x + radius), self.heatmap_size[0])
            end_y = min(int(y + radius), self.heatmap_size[1])

            for pos_x in range(start_x, end_x):
                for pos_y in range(start_y, end_y):
                    target[i * 2, pos_y, pos_x] = x  #- 0.5
                    target[i * 2 + 1, pos_y, pos_x] = y  #- 0.5
                    target_weight[i * 2, pos_y, pos_x] = 1. / self.heatmap_size[0]
                    target_weight[i * 2 + 1, pos_y, pos_x] = 1. / self.heatmap_size[1]

        return target, target_weight

    def neighbor_field(self, heatmaps):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        group_idx = [[0, 1, 2, 3, 4], [5, 7, 9], [6, 8, 10], [5, 6], [11, 12], [11, 13, 15], [12, 14, 16]]
        target = np.zeros((self.num_joints,
                            self.heatmap_size[1],
                            self.heatmap_size[0]),
                            dtype=np.float32)

        binary_hm = (heatmaps > 0.3).astype(np.float32)
        for g_idx in group_idx:
            selected_hm = binary_hm[g_idx]
            maxm_hm = np.max(selected_hm, axis=0, keepdims=True)
            maxm_hm = np.repeat(maxm_hm, len(g_idx), axis=0)
            np.maximum(target[g_idx], maxm_hm, out=target[g_idx])

        return target
