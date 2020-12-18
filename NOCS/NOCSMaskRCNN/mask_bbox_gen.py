"""
Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation
Detection and evaluation

Modified based on Mask R-CNN(https://github.com/matterport/Mask_RCNN)
Written by He Wang
"""

import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='detect', type=str, help="detect/eval")
parser.add_argument('--use_regression', dest='use_regression', action='store_true')
parser.add_argument('--use_delta', dest='use_delta', action='store_true')
parser.add_argument('--ckpt_path', type=str, default='logs/nocs_rcnn_res50_bin32.h5')
parser.add_argument('--data', type=str, help="val/real_test", default='real_test')
parser.add_argument('--outdir', type=str, help="output path")
parser.add_argument('--gpu',  default='0', type=str)
parser.add_argument('--draw', dest='draw', action='store_true', help="whether draw and save detection visualization")
parser.add_argument('--num_eval', type=int, default=-1)

parser.set_defaults(use_regression=False)
parser.set_defaults(draw=False)
parser.set_defaults(use_delta=False)
args = parser.parse_args()

mode = args.mode
data = args.data
outdir = args.outdir
ckpt_path = args.ckpt_path
use_regression = args.use_regression
use_delta = args.use_delta
num_eval = args.num_eval

os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
print('Using GPU {}.'.format(args.gpu))

import sys
import datetime
import glob
import time
import numpy as np
import utils
import model as modellib
import _pickle as cPickle
from maskRCNNConfig import NOCSInferenceRCNNConfig
import cv2
import matplotlib.pyplot as plt
import ruamel.yaml as yaml

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to COCO trained weights
COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")



if __name__ == '__main__':
    if outdir == None:
        print("Please provide --outdir property")
        exit()

    config = NOCSInferenceRCNNConfig()
    config.display()

    # Training dataset
    # dataset directories
    camera_dir = os.path.join('data', 'camera')
    real_dir = os.path.join('data', 'real')
    coco_dir = os.path.join('data', 'coco')

    #  real classes
    coco_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                  'bus', 'train', 'truck', 'boat', 'traffic light',
                  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                  'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                  'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                  'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                  'kite', 'baseball bat', 'baseball glove', 'skateboard',
                  'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                  'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                  'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                  'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                  'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                  'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                  'teddy bear', 'hair drier', 'toothbrush']

    
    synset_names = ['BG', #0
                    'bottle', #1
                    'bowl', #2
                    'camera', #3
                    'can',  #4
                    'laptop',#5
                    'mug'#6
                    ]

    class_map = {
        'bottle': 'bottle',
        'bowl':'bowl',
        'cup':'mug',
        'laptop': 'laptop',
    }


    coco_cls_ids = []
    for coco_cls in class_map:
        ind = coco_names.index(coco_cls)
        coco_cls_ids.append(ind)
    config.display()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                                config=config,
                                model_dir=MODEL_DIR)
    

    # Load trained weights (fill in path to trained weights here)
    model_path = ckpt_path
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    elapse_times = []
    print('*'*50)
    image_start = time.time()

    # record results
    result = {}

    # load all scenes and images
    scenes = []
    files_in_folder = os.listdir(data)
    for file_name in files_in_folder:
        full_path = os.path.join(data, file_name)
        if os.path.isdir(full_path):
            scenes.append(file_name)
    
    print (scenes)

    for scene in scenes:
        img_paths = []
        output_yaml = {}
        color_maps = os.path.join(data, scene, "*_color.png")
        img_paths.extend(glob.glob(color_maps))
        bad_maps = os.path.join(data, scene, "*_bad.png")
        img_paths.extend(glob.glob(bad_maps))

        # export yaml path
        export_path = os.path.join(outdir, scene)
        yaml_file_path = os.path.join(export_path, "export.yml")
        if not os.path.exists(export_path):
            os.makedirs(export_path)

        # deal with each image
        for img_path in img_paths:
            print ("scene {}, image {} is being processed".format(scene, img_path))

            # get pic id
            imgname = os.path.basename(img_path)
            # get id
            imgid = int(imgname[:imgname.find('_')])
            # export yaml data
            yaml_item = []

            if (imgid in output_yaml):
                print ("Warning: the id {} has already been added".format(imgid))

            # read images
            img = cv2.imread(img_path)[:,:,::-1]    # from BGR to RGB
            # feed into the model
            detect_result = model.detect([img], verbose=0)
            r = detect_result[0]
            # get rois, class_id and scores
            obj_n = len(r['class_ids'])
            print ("total {} objects are detected.".format(obj_n))
            for i in range(obj_n):
                yaml_item.append({
                    "obj_bb": r['rois'][i].tolist(),
                    "obj_id": int(r['class_ids'][i]),
                    "score": float(r['scores'][i])
                })
                # write out images
                output_mask = np.where(r['masks'][:,:,i]==0,255,0)
                cv2.imwrite(os.path.join(export_path, ("{}_{}_{}_mask.png").format(imgid, i+1, r['class_ids'][i])),output_mask)
            output_yaml[imgid] = yaml_item

        # output yml file.
        with open(yaml_file_path, "w", encoding="utf-8") as f:
            yaml.dump(output_yaml, f)
            print("saved data to {}".format(yaml_file_path))
        