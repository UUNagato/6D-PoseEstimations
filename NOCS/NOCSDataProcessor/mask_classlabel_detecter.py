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
parser.add_argument('--gpu',  default='0', type=str)
parser.add_argument('--draw', dest='draw', action='store_true', help="whether draw and save detection visualization")
parser.add_argument('--num_eval', type=int, default=-1)

parser.set_defaults(use_regression=False)
parser.set_defaults(draw=False)
parser.set_defaults(use_delta=False)
args = parser.parse_args()

mode = args.mode
data = args.data
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
import skimage
import matplotlib.pyplot as plt
import cv2

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to COCO trained weights
COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")

TEST_IMG_PATH = './data/real/test/scene_1/*_color.png'


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
    img_paths = glob.glob(TEST_IMG_PATH)
    print("total %d images are prepared" % len(img_paths))
    images = []
    img_n = min(len(img_paths), 1)
    print("total %d images are going to be read" % img_n)
    for i in range(img_n):
        image = cv2.imread(img_paths[i])[:,:,::-1] # convert BGR to RGB
        images.append(image)

    image = cv2.imread("./data/real/test/scene_3/0000_color.png")[:,:,::-1]
    images.append(image)

    ## detection
    #print ('input image shape:{}, dtype{}'.format(image.shape, image.dtype))
    plt.imshow(images[1])
    plt.show()

    start = time.time()
    detect_result = model.detect(images, verbose=0)
    print (detect_result[0]['scores'])
    print (detect_result[1]['scores'])
    print("result number:%d" % len(detect_result))
    r = detect_result[1]
    elapsed = time.time() - start

    # data check
    
    print('\nDetection takes {:03f}s.'.format(elapsed))
    result['pred_class_ids'] = r['class_ids']
    result['pred_bboxes'] = r['rois']
    result['pred_RTs'] = None   
    result['pred_scores'] = r['scores']

    print ('deteced {} classes'.format(len(r['class_ids'])))
    print ('mask dtype:{}'.format(r['masks'].dtype))
    #for i in range(len(r['class_ids'])):
    #    print('object class id:{}'.format(r['class_ids'][i]))
    #    print('2dbox:{}'.format(r['rois'][i]))
    for i in range(2):
        lr = detect_result[i]
        bbox = lr['rois'][0]
        plt.imshow(lr['masks'][:,:,0], cmap=plt.cm.get_cmap('binary'))
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
