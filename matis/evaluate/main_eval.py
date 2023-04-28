import json
import numpy as np
import argparse
from tqdm import tqdm

from matis.evaluate.compute_all_iou import eval_segmentation
from matis.evaluate.compute_all_metrics import eval_classification


def load_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data

def save_json(data, json_file, indent=4):
    with open(json_file, "w") as f:
        json.dump(data, f, indent=indent)


def eval_task(task, coco_anns, preds, features, version):
    if task=='recognition':
        m0, m1, m2 = eval_classification(preds)
    elif task == 'tools':
        m0, m1, m2 = eval_segmentation(preds, features, coco_anns, version)
    else:
        raise('Unknown task')
    return (m0, m1, m2)

def main(coco_ann_path, pred_path, tasks=['recognition', 'tools'], features=None, fold=None, version=None, visualization=False):
    # Load coco anns and preds
    coco_anns = load_json(coco_ann_path)
    preds = load_json(pred_path)
    all_metrics = {}
    for task in tasks:
        task_metric = eval_task(task, coco_anns, preds, features, version)
        all_metrics[task] = task_metric
        print('{} task mAP: {}'.format(task, task_metric[0]))
    overall_mAP = np.mean([u[0] for u in list(all_metrics.values())])
    print('Overall mAP: {}'.format(overall_mAP))
    return task_metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation parser')
    parser.add_argument('--coco-ann-path', default=None,
                        type=str, help='path to coco style anotations')
    parser.add_argument('--coco-pred-path', default=None,
                        type=str, help='path to predictions')
    parser.add_argument('--tasks', nargs='+', help='tasks to be evaluated',
                        required=True, default=None)
    parser.add_argument('--visualization', default=False, action='store_true')

    save_path = ''
    args = parser.parse_args()
    print(args)
    main(args.coco_ann_path, args.coco_pred_path, args.tasks, args.visualization)