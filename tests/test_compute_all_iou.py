import importlib.util
import os

import numpy as np
import pycocotools.mask as m
import skimage.io as io
import torch

_PATH = os.path.join(os.path.dirname(__file__), '..', 'matis', 'evaluate', 'compute_all_iou.py')
_spec = importlib.util.spec_from_file_location('compute_all_iou', _PATH)
compute_all_iou = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compute_all_iou)


def _run_eval(tmp_path, monkeypatch, pred_label):
    # one frame, GT contains only class 1, prediction is a perfect class-1 mask
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('PYTHONBREAKPOINT', '0')
    file_name = 'seq_1_frame000.png'
    gt_dir = tmp_path / 'data' / 'endovis_2017' / 'annotations' / 'images'
    gt_dir.mkdir(parents=True)
    gt = np.zeros((1024, 1280), dtype=np.uint8)
    gt[100:200, 100:300] = 1
    io.imsave(str(gt_dir / file_name), gt, check_contrast=False)

    rle = m.encode(np.asfortranarray((gt == 1).astype(np.uint8)))
    features = tmp_path / 'features.pth'
    torch.save({'features': [{'file_name': file_name, 'segments': {'100 100 300 200': rle}}]}, str(features))
    data = {'images': [{'id': 1, 'file_name': file_name}]}
    prob_tools = [0.0] * 7
    prob_tools[pred_label - 1] = 1.0
    preds = {file_name: {'bboxes': [{'bbox': [100, 100, 300, 200], 'prob_tools': prob_tools}]}}
    return compute_all_iou.eval_segmentation(preds, str(features), data, 'endovis_2017')


def test_absent_classes_are_ignored_in_mciou(tmp_path, monkeypatch):
    # 1 perfectly segmented class + 6 classes with no GT and no predictions
    total_gt_iou, total_iou, total_ciou = _run_eval(tmp_path, monkeypatch, pred_label=1)
    assert total_gt_iou == 1.0
    assert total_iou == 1.0
    assert np.isclose(total_ciou, 1.0)  # was 1/7 = 0.143 before the fix


def test_false_positive_class_still_counts_as_zero(tmp_path, monkeypatch):
    # class 1 missed (IoU 0) and class 2 predicted but absent in GT (IoU 0)
    total_gt_iou, total_iou, total_ciou = _run_eval(tmp_path, monkeypatch, pred_label=2)
    assert total_gt_iou == 0.0
    assert total_iou == 0.0
    assert total_ciou == 0.0
