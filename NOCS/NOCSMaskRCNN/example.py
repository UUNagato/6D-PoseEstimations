"""
Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation
Detection and evaluation

Modified based on Mask R-CNN(https://github.com/matterport/Mask_RCNN)
Written by He Wang
"""

import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', type=str, default='logs/nocs_rcnn_res50_bin32.h5')
parser.add_argument('--data', type=str, help="val/real_test", default='real_test')
parser.add_argument('--gpu',  default='0', type=str)

args = parser.parse_args()

ckpt_path = args.ckpt_path
data = args.data
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
print('Using GPU {}.'.format(args.gpu))

import sys
import datetime
import glob
import time
import numpy as np
import model as modellib
import _pickle as cPickle
from maskRCNNConfig import NOCSInferenceRCNNConfig
import skimage
import matplotlib.pyplot as plt
import cv2

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to COCO trained weights
COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")

TEST_IMG_PATH = './0000_color.png'


if __name__ == '__main__':

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

    if data in ['real_train', 'real_test']:
        intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    else: ## CAMERA data
        intrinsics = np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]])

    elapse_times = []
    print('*'*50)
    image_start = time.time()

    # record results
    result = {}

            # loading ground truth
    image = cv2.imread(TEST_IMG_PATH)[:,:,::-1] # convert BGR to RGB

    ## detection
    print ('input image shape:{}, dtype{}'.format(image.shape, image.dtype))
    plt.imshow(image)
    plt.show()
    start = time.time()
    detect_result = model.detect([image], verbose=0)
    r = detect_result[0]
    elapsed = time.time() - start
    
    print('\nDetection takes {:03f}s.'.format(elapsed))
    result['pred_class_ids'] = r['class_ids']
    result['pred_bboxes'] = r['rois']
    result['pred_RTs'] = None   
    result['pred_scores'] = r['scores']

    print ('deteced {} classes'.format(len(r['class_ids'])))
    print ('mask dtype:{}'.format(r['masks'].dtype))
    for i in range(len(r['class_ids'])):
        print('object class id:{}'.format(r['class_ids'][i]))
        print('2dbox:{}'.format(r['rois'][i]))
        bbox = r['rois'][i]
        plt.imshow(r['masks'][:,:,i], cmap=plt.cm.get_cmap('binary'))
        plt.gca().add_patch(plt.Rectangle((bbox[1], bbox[0]), bbox[3]-bbox[1], bbox[2]-bbox[0],fill=False))
        plt.show()



    
    if len(r['class_ids']) == 0:
        print('No instance is detected.')

    # print('Aligning predictions...')
    # start = time.time()
    ''' To run this part, you will need to read in depth map
    result['pred_RTs'], result['pred_scales'], error_message, elapses =  utils.align(r['class_ids'], 
                                                                                    r['masks'], 
                                                                                    r['coords'], 
                                                                                    depth, 
                                                                                    intrinsics, 
                                                                                    synset_names, 
                                                                                    image_path)
                                                                                    #save_dir+'/'+'{}_{}_{}_pred_'.format(data, image_path_parsing[-2], image_path_parsing[-1]))
    '''
