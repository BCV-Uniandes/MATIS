#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from bdb import Breakpoint
import os
import torch
import logging
import numpy as np
import traceback

from copy import deepcopy
from . import endovis_2017_helper as ava_helper
from . import cv2_transform as cv2_transform
from . import utils as utils
from .build import DATASET_REGISTRY

logger = logging.getLogger(__name__)


@DATASET_REGISTRY.register()
class Endovis_2017(torch.utils.data.Dataset):
    """
    SAR-RARP50 Dataset
    """

    def __init__(self, cfg, split):
        self.cfg = cfg
        self._split = split
        self._sample_rate = cfg.DATA.SAMPLING_RATE
        self._video_length = cfg.DATA.NUM_FRAMES
        self._seq_len = self._video_length * self._sample_rate
        self._num_classes = {key: n_class for key, n_class in \
                            zip(cfg.TASKS.TASKS, cfg.TASKS.NUM_CLASSES)}
        # Augmentation params.
        self._data_mean = cfg.DATA.MEAN
        self._data_std = cfg.DATA.STD
        self._use_bgr = cfg.AVA.BGR
        self.random_horizontal_flip = cfg.DATA.RANDOM_FLIP
        if self._split == "train":
            self._crop_size = cfg.DATA.TRAIN_CROP_SIZE
            self._jitter_min_scale = cfg.DATA.TRAIN_JITTER_SCALES[0]
            self._jitter_max_scale = cfg.DATA.TRAIN_JITTER_SCALES[1]
            self._use_color_augmentation = cfg.AVA.TRAIN_USE_COLOR_AUGMENTATION
            self._pca_jitter_only = cfg.AVA.TRAIN_PCA_JITTER_ONLY
            self._pca_eigval = cfg.DATA.TRAIN_PCA_EIGVAL
            self._pca_eigvec = cfg.DATA.TRAIN_PCA_EIGVEC
        else:
            self._crop_size = cfg.DATA.TEST_CROP_SIZE
            self._test_force_flip = cfg.AVA.TEST_FORCE_FLIP

        # Read Mask2Former features
        if cfg.MASKFORMER.ENABLE:
            self.feature_boxes = ava_helper.load_features_boxes(cfg)
        else: 
            self.features_boxes = None
        
        self._load_data(cfg)

    def _load_data(self, cfg):
        """
        Load frame paths and annotations from files

        Args:
            cfg (CfgNode): config
        """
        # Loading frame paths.
        (
            self._image_paths,
            self._video_idx_to_name,
        ) = ava_helper.load_image_lists(cfg, is_train=(self._split == "train"))

        # Loading annotations for boxes and labels.
        boxes_and_labels = ava_helper.load_boxes_and_labels(
            cfg, mode=self._split
        )

        assert len(boxes_and_labels) == len(self._image_paths), '{} & {}'.format(len(boxes_and_labels),len(self._image_paths))
        boxes_and_labels = [
            boxes_and_labels[self._video_idx_to_name[i]]
            for i in range(len(self._image_paths))
        ]

        # Get indices of keyframes and corresponding boxes and labels.
        (
            self._keyframe_indices,
            self._keyframe_boxes_and_labels,
        ) = ava_helper.get_keyframe_data(boxes_and_labels)
        # Calculate the number of used boxes.
        self._num_boxes_used = ava_helper.get_num_boxes_used(
            self._keyframe_indices, self._keyframe_boxes_and_labels
        )

        self.print_summary()

    def print_summary(self):
        logger.info("=== ENDOVIS 2017 {} dataset summary ===".format(self.cfg.FOLD))
        logger.info("Split: {}".format(self._split))
        logger.info("Number of videos: {}".format(len(self._image_paths)))
        total_frames = sum(
            len(video_img_paths) for video_img_paths in self._image_paths
        )
        logger.info("Number of frames: {}".format(total_frames))
        logger.info("Number of key frames: {}".format(len(self)))
        logger.info("Number of boxes: {}.".format(self._num_boxes_used))

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._keyframe_indices)

    def _images_and_boxes_preprocessing_cv2(self, imgs, boxes):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip with opencv as backend.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """
        height, width, _ = imgs[0].shape

        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        boxes = cv2_transform.clip_boxes_to_image(boxes, height, width)
        boxes = [boxes]

        # The image now is in HWC, BGR format.
        if self._split == "train" and not self.cfg.DATA.JUST_CENTER:
            imgs, boxes = cv2_transform.random_short_side_scale_jitter_list(
                imgs,
                min_size=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
            )
            imgs, boxes = cv2_transform.random_crop_list(
                imgs, self._crop_size, order="HWC", boxes=boxes
            )

            if self.random_horizontal_flip:
                # random flip
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    0.5, imgs, order="HWC", boxes=boxes
                )
        elif self._split == "val" or self.cfg.DATA.JUST_CENTER:
            # Short side to test_scale. Non-local and STRG uses 256.
            imgs = [cv2_transform.scale(self._crop_size, img) for img in imgs]
            boxes = [
                cv2_transform.scale_boxes(
                    self._crop_size, boxes[0], height, width
                )
            ]
            imgs, boxes = cv2_transform.spatial_shift_crop_list(
                self._crop_size, imgs, 1, boxes=boxes
            )

        else:
            raise NotImplementedError(
                "Unsupported split mode {}".format(self._split)
            )

        # Convert image to CHW keeping BGR order.
        imgs = [cv2_transform.HWC2CHW(img) for img in imgs]

        # Image [0, 255] -> [0, 1].
        imgs = [img / 255.0 for img in imgs]

        imgs = [
            np.ascontiguousarray(
                # img.reshape((3, self._crop_size, self._crop_size))
                img.reshape((3, imgs[0].shape[1], imgs[0].shape[2]))
            ).astype(np.float32)
            for img in imgs
        ]

        # Do color augmentation (after divided by 255.0).
        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = cv2_transform.color_jitter_list(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
                )

            imgs = cv2_transform.lighting_list(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = [
            cv2_transform.color_normalization(
                img,
                np.array(self._data_mean, dtype=np.float32),
                np.array(self._data_std, dtype=np.float32),
            )
            for img in imgs
        ]

        # Concat list of images to single ndarray.
        imgs = np.concatenate(
            [np.expand_dims(img, axis=1) for img in imgs], axis=1
        )

        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            imgs = imgs[::-1, ...]

        imgs = np.ascontiguousarray(imgs)
        imgs = torch.from_numpy(imgs)
        boxes = cv2_transform.clip_boxes_to_image(
            boxes[0], imgs[0].shape[1], imgs[0].shape[2]
        )
        return imgs, boxes

    def __getitem__(self, idx):
        """
        Generate corresponding clips, boxes, labels and metadata for given idx.

        Args:
            idx (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (ndarray): the label for correspond boxes for the current video.
            idx (int): the video index provided by the pytorch sampler.
            extra_data (dict): a dict containing extra data fields, like "boxes",
                "ori_boxes" and "metadata".
        """
        video_idx, sec_idx, sec, center_idx = self._keyframe_indices[idx]
        video_name = self._video_idx_to_name[video_idx]
        folder_to_images = "/".join(self._image_paths[video_idx][0].split('/')[:-1])
        complete_name = video_name+'_frame'+str(sec).zfill(3)+'.png'
        path_complete_name = os.path.join(folder_to_images,complete_name)
        assert path_complete_name==self._image_paths[video_idx][sec], path_complete_name + ' & ' + self._image_paths[video_idx][sec]
        center_idx = self._image_paths[video_idx].index(path_complete_name)
        assert center_idx==sec

        # Get the frame idxs for current clip.
        seq = utils.get_sequence(
            center_idx,
            self._seq_len // 2,
            self._sample_rate,
            num_frames=len(self._image_paths[video_idx]),
        )
        clip_label_list = deepcopy(self._keyframe_boxes_and_labels[video_idx][sec_idx])
        if len(clip_label_list) == 0:
            print(complete_name)

        # Get boxes and labels for current clip.
        boxes = []

        # Tasks to solve in training.
        all_tasks = self.cfg.TASKS.TASKS
        all_labels = {t:[-1 for _ in clip_label_list] for t in all_tasks}

        if self.cfg.MASKFORMER.ENABLE:
            faster_features = []
            # Faster features 
            try:
                box_features = self.feature_boxes[complete_name] 
            except:
                traceback.print_exc()
                import pdb; pdb.set_trace()

        else:
            faster_features = None
        
        boxes_idxs = []
        if len(clip_label_list)>0:
            for b_idx, box_labels in enumerate(clip_label_list):
                boxes.append(box_labels[0])
                boxes_idxs.append(3)
                if self.cfg.MASKFORMER.ENABLE:
                    faster_box_key = " ".join(map(str,box_labels[0]))

                    if box_labels[0]!=[0,0,0,0]:
                        assert faster_box_key in box_features, 'The box {} is not prsent {},{}'.format(faster_box_key,video_name,sec)
                        faster_features.append(np.array(box_features[faster_box_key]))
                    else:
                        faster_features.append(np.zeros(256))

                for task in all_tasks:
                    if task=='recognition':
                        if b_idx>0:
                            assert all_labels[task]==list(box_labels[2]), '{} {}'.format(all_labels[task],list(box_labels[2]))
                        all_labels[task]= list(box_labels[2])
                    elif task == 'tools' or task=='segmentation':
                        all_labels[task][b_idx] = box_labels[1]
                    else:
                        raise ValueError('{} is not a valid task'.format(task))
        else:
            breakpoint()

        ori_boxes = np.array(boxes)[:,:4].copy()
            
        # Score is not used.
        boxes = np.array(boxes)[:, :4]
        
        # Load images of current clip.
        image_paths = [self._image_paths[video_idx][frame] for frame in seq]
        imgs = utils.retry_load_images(
            image_paths, backend=self.cfg.AVA.IMG_PROC_BACKEND
        )

        # Preprocess images and boxes
        imgs, boxes = self._images_and_boxes_preprocessing_cv2(
            imgs, boxes=boxes
        )

        max_boxes = self.cfg.DATA.MAX_BBOXES
        bbox_mask = np.zeros(max_boxes,dtype=bool)
        bbox_mask[:len(boxes)] = True
        if len(boxes)<max_boxes:
            c_boxes = np.concatenate((boxes,np.zeros((max_boxes-len(boxes),4))),axis=0)
            c_faster_features = np.concatenate((faster_features,np.zeros((max_boxes-len(faster_features),256))),axis=0)
            boxes = c_boxes
            faster_features = c_faster_features
        
        imgs = utils.pack_pathway_output(self.cfg, imgs)

        return imgs, all_labels, complete_name, boxes, ori_boxes, bbox_mask, faster_features