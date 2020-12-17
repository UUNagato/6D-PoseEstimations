# NOCS data prepocessing
import numpy as np
import argparse
import os
import glob
import cv2
from skimage import exposure
import matplotlib.pyplot as plt
import utils # from NOCS
from NOCS_pose_error import decompose_transform_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--path', help='Path to the dataset', required=True)
parser.add_argument('--augmented', action='store_true')
parser.add_argument('--crop', help='whether we should crop the image', action='store_true')
parser.add_argument('--dataset', help='val/real_test', default='val')
parser.add_argument('--no_uniformly_scaling', help='If specified, the cropped image may use a non uniformly scaling', action='store_true')
parser.add_argument('--output', help='The output path', default='.')
parser.add_argument('--debug', help='debug mode', action='store_true')
parser.add_argument('--compressed', help='compress the output file', action='store_true')

args = parser.parse_args()

synset_names = ['BG', #0
                    'bottle', #1
                    'bowl', #2
                    'camera', #3
                    'can',  #4
                    'laptop',#5
                    'mug'#6
                    ]

intrinsics = np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]])

def debug_showimg(patches, title):
    amount = len(patches)
    if amount > 10:
        show_imgs = patches[:amount]
    else:
        show_imgs = patches
    
    final_img = np.hstack(tuple(show_imgs))
    plt.imshow(final_img)
    plt.title(title)
    plt.show()

def augment_image(image, mask_im, coord_map):
    # apply random gamma correction
    gamma = np.random.uniform(0.8, 1)
    gain = np.random.uniform(0.8, 1)
    image = exposure.adjust_gamma(image, gamma, gain)

    rotate_degree = np.random.uniform(-5, 5)
    image, mask_im, coord_map = utils.rotate_and_crop_images(image, masks=mask_im, coords=coord_map, rotate_degree=rotate_degree)
    
    return image, mask_im, coord_map

def fit_cropped_box_ratio(cropped_box, image_size, target_ratio):
    if args.no_uniformly_scaling:
        return cropped_box
    
    cropped_h = cropped_box[2] - cropped_box[0]
    cropped_w = cropped_box[3] - cropped_box[1]
    cropped_ratio = cropped_h / cropped_w
    if cropped_ratio < target_ratio:
        cropped_h_new = round(cropped_w * target_ratio)
        # extend it
        cropped_box[2] = cropped_box[0] + cropped_h_new
        # compute an offset
        # here we assume the mask is much smaller than the whole image, so it will always fit in the image.
        offset = (cropped_h_new - cropped_h) // 2
        offset = max(max(0, cropped_box[2] - image_size[0]), offset)    # if it's out of bound, we need to offset more
        offset = min(offset, cropped_box[0])        # if the offset makes the bbox minimum edge less than 0, fix it.
        cropped_box[0] -= offset
        cropped_box[2] -= offset
    elif cropped_ratio > target_ratio:
        cropped_w_new = round(cropped_h / target_ratio)
        cropped_box[3] = cropped_box[1] + cropped_w_new

        offset = (cropped_w_new - cropped_w) // 2
        offset = max(max(0, cropped_box[3] - image_size[1]), offset)    # if it's out of bound, we need to offset more
        offset = min(offset, cropped_box[1])        # if the offset makes the bbox minimum edge less than 0, fix it.
        cropped_box[1] -= offset
        cropped_box[3] -= offset

    
    #assert cropped_box[0] >= 0 and cropped_box[1] >= 0, "{}".format(cropped_box)
    #assert cropped_box[2] <= image_size[0] and cropped_box[3] <= image_size[1], "{},{}".format(cropped_box, image_size)

    return cropped_box

        


def load_gt_data(folderpath:str, imgpath:str, img_id:str):
    if not img_id.isdigit():
        print ("An unexpected img_id {} is feed into load_gt_data.".format(img_id))
        return None, None, None, None, None, None
    meta_name = img_id + '_meta.txt'
    meta_path = os.path.join(folderpath, meta_name)
    if not os.path.exists(meta_path):
        print ("Warning: the meta file:{} cannot be found, so this image is skipped".format(meta_path))
        return None, None, None, None, None, None
    
    inst_dict = {}
    obj_dict = {}
    try:
        with open(meta_path, 'r') as f:
            for line in f:
                line_info = line.split(' ')
                inst_id = int(line_info[0])  ##one-indexed
                cls_id = int(line_info[1])  ##zero-indexed

                if len(line_info) == 4:
                    obj_path = line_info[2] + "/" + line_info[3]
                    if obj_path[-1] == '\n':
                        obj_path = obj_path[:-1]
                else:
                    # TODO, for real set
                    obj_path = ""
                # skip background objs
                # symmetry_id = int(line_info[2])
                inst_dict[inst_id] = cls_id
                obj_dict[inst_id] = obj_path
    except Exception as e:
        print ("Failed with IO of meta file:{}".format(meta_path))
        return None, None, None, None, None, None
    
    # read image, coord (for transformation), mask
    base_path = os.path.join(folderpath, img_id)
    mask_path = base_path + '_mask.png'
    coord_path = base_path + '_coord.png'

    image = cv2.imread(imgpath)[:,:,:3]
    image = image[..., ::-1]    # from BGR to RGB

    # if grayscale, convert to RGB for consistency.
    if image.ndim != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    mask_im = cv2.imread(mask_path)[:, :, 2]        # use R channel
    coord_map = cv2.imread(coord_path)[:, :, :3]
    coord_map = coord_map[:, :, ::-1]

    if args.augmented:
        # if augmented, apply random rotation
        image, mask_im, coord_map = augment_image(image, mask_im, coord_map)
    
    # process data
    cdata = np.array(mask_im, dtype=np.int32)

    instance_ids = list(np.unique(cdata))
    instance_ids = sorted(instance_ids)
    if instance_ids[-1] != 255:
        print ("A wrong sample is detected, and thus skipped. {}".format(imgpath))
        return None, None, None, None, None, None
    del instance_ids[-1]        # remove the background id 255.
    
    cdata[cdata==255] = -1
    assert(np.unique(cdata).shape[0] < 20)  # no more than 20 instances in a single image

    num_instance = len(instance_ids)
    h, w = cdata.shape

    coord_map = np.array(coord_map, dtype=np.float32) / 255
    coord_map[:, :, 2] = 1 - coord_map[:, :, 2]

    masks = np.zeros([h, w, num_instance], dtype=np.uint8)
    coords = np.zeros((h, w, num_instance, 3), dtype=np.float32)
    class_ids = np.zeros([num_instance], dtype=np.int_)
    
    inst_id_to_be_deleted = []
    for inst_id in inst_dict.keys():
        if inst_dict[inst_id] == 0 or (not inst_id in instance_ids):
            inst_id_to_be_deleted.append(inst_id)
    for delete_id in inst_id_to_be_deleted:
        del inst_dict[delete_id]
        del obj_dict[delete_id]
    
    i = 0
    for inst_id in instance_ids:
        if not inst_id in inst_dict:
            continue
        inst_mask = np.equal(cdata, inst_id)
        assert np.sum(inst_mask) > 0
        assert inst_dict[inst_id]

        masks[:, :, i] = inst_mask
        coords[:, :, i, :] = np.multiply(coord_map, np.expand_dims(inst_mask, axis=-1))

        class_ids[i] = inst_dict[inst_id]
        i += 1
    
    masks = masks[:, :, :i]
    coords = coords[:, :, :i, :]
    coords = np.clip(coords, 0, 1)

    class_ids = class_ids[:i]

    # ====================================================================================
    # compute bbox and evaluate gt pose by matching coords to NOCS coords.
    # ====================================================================================
    bbox = utils.extract_bboxes(masks)

    # load depth
    depth_path = base_path + "_depth.png"
    depth = cv2.imread(depth_path, -1)

    if len(depth.shape) == 3:
        # This is encoded depth image, let's convert
        depth16 = np.uint16(depth[:, :, 1]*256) + np.uint16(depth[:, :, 2]) # NOTE: RGB is actually BGR in opencv
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        depth16 = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'
    
    RTs, _, error_message, _ = utils.align(class_ids, masks, coords, depth16, intrinsics, synset_names, imgpath)

    if len(error_message):
       print ("[ Error ]: Error {} happened when aligning the coordinates".format(error_message))
       return None, None, None, None, None, None

    # ======================================================================================
    # cropped the images and decompose the RTs
    # ======================================================================================
    image_patches = []
    if (args.crop):
        cropped_size = (128, 128)
        mask_h, mask_w = masks.shape[0:2]

        target_ratio = cropped_size[0] / cropped_size[1]
        for i in range(masks.shape[2]):
            # add a random extension to bbox, so that the object won't fill the whole image patch
            extend = np.random.randint(1, 10, 4)

            cropped_box = bbox[i].copy()
            cropped_box[0] = max(0, cropped_box[0] - extend[0])
            cropped_box[1] = max(0, cropped_box[1] - extend[1])
            cropped_box[2] = min(mask_h, cropped_box[2] + extend[2])
            cropped_box[3] = min(mask_w, cropped_box[3] + extend[3])

            # if uniformly scaling, we need to make the patch the same ratio with cropped size
            cropped_box = fit_cropped_box_ratio(cropped_box, (mask_h, mask_w), target_ratio)

            # if a bad bbox is detected, skip this image
            if cropped_box[0] < 0 or cropped_box[1] < 0 or cropped_box[2] > mask_h or cropped_box[3] > mask_w:
                print ("A bad image {} is skipped".format(imgpath))
                return None, None, None, None, None, None

            # crop the image
            patch_img = image[cropped_box[0]:cropped_box[2], cropped_box[1]:cropped_box[3]]
            patch_img = cv2.resize(patch_img, cropped_size, interpolation=cv2.INTER_LINEAR)

            image_patches.append(patch_img)
        
        if len(image_patches) == 0:
            print ("An empty image {} is skipped".format(imgpath))
            return None, None, None, None, None, None

    # debug, show the patches.    
    if args.debug:
        print (RTs)
        print (inst_dict)
        print (obj_dict)
        debug_showimg(image_patches, imgpath)
    
    return image_patches, masks, RTs, inst_dict, obj_dict, bbox

def prepare_output(output_path, synset_names):
    class_num = len(synset_names)
    os.makedirs(output_path, exist_ok=True)
    if (args.crop):
        return [{"bgr_y_src":[], "matrix_rot_y":[], "matrix_tra_y":[], "scale_y":[], "obj_list":[]} \
                    for i in range(class_num)]

    return [{"scene_id":[], "image_id":[], "class_id":[], "bbox":[], "matrix_rot_y":[], "matrix_tra_y":[], "scale_y":[], "obj_list":[]} \
                    for i in range(class_num)]


def process_CAMERA(path):
    # find folders in the scene
    scene_list = []
    for item in os.listdir(path):
        scene_path = os.path.join(path, item)
        if os.path.isdir(scene_path):
            scene_list.append(item)
    
    scene_amount = len(scene_list)
    img_amount = 0
    img_processed_amount = 0
    print ("Found {} scene folders in total".format(scene_amount))

    output_list = prepare_output(args.output, synset_names)
    if os.path.isdir(args.output) == False:
        print ("Cannot find or create output folder:{}".format(args.output))
        return
    
    # iterate each folder
    for scene in scene_list:
        folder_path = os.path.join(path, scene)
        # find all color pngs
        png_files = os.path.join(folder_path, "./*_color.png")    # ./ is for compatibility to Windows, although Windows is not recommended
        png_files = glob.glob(png_files)

        for img_path in png_files:
            # try to split image number
            img_name = os.path.split(img_path)[-1]
            img_id = img_name.split('_')[0]
            patches, masks, RTs, inst_dict, obj_dict, bbox = load_gt_data(folder_path, img_path, img_id)

            if patches is None:
                continue

            assert len(inst_dict) == masks.shape[-1]
            if args.crop:
                assert masks.shape[-1] == len(patches)
            assert RTs.shape[0] == masks.shape[-1]

            inst_number = len(inst_dict)
            inst_keys = list(inst_dict.keys())
            # save the result
            for i in range(inst_number):
                cls_id = inst_dict[inst_keys[i]]
                RT = RTs[i]
                T, S, R = decompose_transform_matrix(RT)
                if (args.crop):
                    output_list[cls_id]['bgr_y_src'].append(patches[i])
                else:
                    output_list[cls_id]['scene_id'].append(int(scene))
                    output_list[cls_id]['image_id'].append(int(img_id))
                    output_list[cls_id]['class_id'].append(cls_id)
                    output_list[cls_id]['bbox'].append(bbox[i])
                output_list[cls_id]['matrix_rot_y'].append(R)
                output_list[cls_id]['matrix_tra_y'].append(T)
                output_list[cls_id]['scale_y'].append(S)
                output_list[cls_id]['obj_list'].append(obj_dict[inst_keys[i]])
    
    # finally, output these value
    save_method = np.savez_compressed if args.compressed else np.savez
    for i in range(len(output_list)):
        output_file_name = synset_names[i] + "_encoder_data"
        if args.compressed:
            output_file_name += "_compressed"
        output_file_path = os.path.join(args.output, output_file_name)

        output_data = output_list[i]
        print ("Saving to file:{}".format(output_file_path))
        try:            # in case the IO exception terminate the program and get all computed data loss
            if args.crop:
                save_method(output_file_path, bgr_y_src=output_data['bgr_y_src'], matrix_rot_y=output_data['matrix_rot_y'], matrix_tra_y=output_data['matrix_tra_y'], \
                        scale_y=output_data['scale_y'], obj_list=output_data['obj_list'])
            else:
                save_method(output_file_path, scene_id=output_data['scene_id'], image_id=output_data['image_id'], class_id=output_data['class_id'], \
                    bbox=output_data['bbox'], matrix_rot_y=output_data['matrix_rot_y'], matrix_tra_y=output_data['matrix_tra_y'], \
                    scale_y=output_data['scale_y'], obj_list=output_data['obj_list'])
        except Exception as e:
            print ("Failed to save to {}".format(output_file_path))
            e.print_exc()

if __name__ == "__main__":
    dataset_path = args.path
    dataset_type = args.dataset

    print ("Current configuration:\npath:{}\ndataset type:{}\noutput path:{}\naugmented:{}\ncropped:{}".format(dataset_path, dataset_type, args.output, args.augmented, args.crop))
    # check path
    if not os.path.exists(dataset_path) or not os.path.isdir(dataset_path):
        print ("The path {} is not a folder".format(dataset_path))
        exit()
    
    if dataset_type == 'val':
        print ("Start process camera dataset")
        process_CAMERA(dataset_path)
    elif dataset_type == 'real_test':
        print ("Sorry, real set is not implemented yet")
        exit()
    else:
        print ("No available dataset type is input")
        exit()