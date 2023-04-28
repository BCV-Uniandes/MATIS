import traceback
import numpy as np
import pycocotools.mask as m
import skimage.io as io
from tqdm import tqdm
import torch
import os
import gc
from copy import copy

def roundbox(box):
    keys = box.split(' ')
    keys = [str(round(float(k),4)) for k in keys]
    return ' '.join(keys)

def floatbox(box):
    keys = box.split(' ')
    return list(map(float,keys))

def boxiou(bb1,bb2):
    x1 = max(bb1[0],bb2[0])
    y1 = max(bb1[1],bb2[1])
    x2 = min(bb1[2],bb2[2])
    y2 = min(bb1[3],bb2[3])
    if x2<x1 or y2<y1:
        return 0.0
    elif y2==y1 and bb1[1]==bb1[3]==bb2[1]==bb2[3]:
        return 1
    elif x2==x1 and bb1[0]==bb1[2]==bb2[0]==bb2[2]:
        return 1
    inter = (x2-x1)*(y2-y1)
    area1 = (bb1[2]-bb1[0])*(bb1[3]-bb1[1])
    area2 = (bb2[2]-bb2[0])*(bb2[3]-bb2[1])
    if (area1+area2-inter)==0:
        breakpoint()
    box_iou = inter/(area1+area2-inter)

    assert box_iou>=0 and box_iou<=1
    return box_iou

def getrealbox(boxes,box):
    maxiou = 0
    realbox = ''
    box = floatbox(box)
    for box2 in boxes:
        iou = boxiou(floatbox(box2),box)
        if iou>maxiou:
            maxiou=iou
            realbox=box2
    
    assert maxiou>0.9 and maxiou<=1
    return realbox

def eval_segmentation(preds,features,data,version):
    cats = {1,2,3,4,5,6,7}
    features = torch.load(features)['features']

    file_dict ={}
    for feat in features:
        try:
            segments = feat['segments']
            file_dict[feat['file_name']]={'file_name': feat['file_name'], 'id': None, 'instances': [], 'categories': [], 'featboxes': segments, 'video_name': None}
        except:
            traceback.print_exc()
            breakpoint()

    try:
        file_name2id = {}
        for d in data['images']:
            file_name = d['file_name'].split('/')[-1]

            file_name2id[d['id']] = file_name
            file_dict[file_name]['id'] = d['id']

            info = file_name.split('_')
            sequence = '_'.join(info[:2])

            file_dict[file_name]['video_name'] = sequence

        data=None
        gc.collect()

        for pred in tqdm(preds,desc='predictions'):
            if pred in file_dict:
                instances = []
                categories = []
                for box in preds[pred]['bboxes']:
                    if box['bbox'] != [0.0, 0.0, 0.0, 0.0]:
                        segment = file_dict[pred]['featboxes'][getrealbox(file_dict[pred]['featboxes'].keys(),'{} {} {} {}'.format(*box['bbox']))]
                        category = np.argmax(box['prob_tools']) 
                        if category < 7:
                            categories.append(category+1)
                            instances.append({'segmentation':segment,'category_id':category+1, 'score': box['prob_tools'][category]})

                instances.sort(key=lambda x: x['score'])

                file_dict[pred]['instances'] = instances
                file_dict[pred]['categories'] = set(categories)

        pred = None
        instances = None
        p_mask = None
        gc.collect()

        ious = []
        gt_ious = []
        pcls_ious = {1:[],2:[],3:[],4:[],5:[],6:[],7:[]}
        for file in tqdm(file_dict):

            if '2018' in version:
                gt_img = io.imread(os.path.join('.','data','endovis_2018','annotations','val', file))
                gt_img[gt_img==6]=4
                gt_img[gt_img==8]=5
                gt_img[gt_img==9]=6
            elif '2017' in version:
                gt_img = io.imread(os.path.join('.','data','endovis_2017','annotations','images', file))

            gt_classes = set(np.unique(gt_img))
            gt_classes.remove(0)

            sem_im = np.zeros((1024,1280))
            for ins in file_dict[file]['instances']:
                p_mask = m.decode(ins['segmentation'])
                sem_im[p_mask==1]=ins['category_id']
            
            categories = set(np.unique(sem_im))
            categories.remove(0)

            class_iou = []
            gt_class_iou = []

            for label in cats:
                if label in gt_classes or label in categories:
                    pred_im = (sem_im==label).astype('uint8')
                    gt_mask = (gt_img==label).astype('uint8')

                    intersection = np.sum(pred_im * gt_mask)
                    union = np.sum(pred_im) + np.sum(gt_mask)
                    im_IoU = intersection/(union-intersection)
                    assert im_IoU>=0 and im_IoU<=1, im_IoU
                    class_iou.append(im_IoU)
                    pcls_ious[label].append(im_IoU)

                    if label in gt_classes and label not in categories:
                        assert im_IoU == 0
                    
                    if label not in gt_classes and label in categories:
                        assert im_IoU == 0
                    
                    if label in gt_classes:
                        gt_class_iou.append(im_IoU)

            assert len(class_iou)==len(gt_classes.union(categories))
            assert len(gt_class_iou)==len(gt_classes)

            if len(class_iou)>0:
                ious.append(float(np.mean(class_iou)))
            
            if len(gt_class_iou)>0:
                gt_ious.append(float(np.mean(gt_class_iou)))

        total_iou = float(np.mean(ious))
        total_gt_iou = float(np.mean(gt_ious))

        for cls in pcls_ious:
            pcls_ious[cls] = float(np.mean(pcls_ious[cls])) if len(pcls_ious[cls])>0 else 0
        total_ciou = float(np.mean(list(pcls_ious.values())))
    except:
        traceback.print_exc()
        breakpoint()

    gc.collect()

    return total_gt_iou, total_iou, total_ciou
