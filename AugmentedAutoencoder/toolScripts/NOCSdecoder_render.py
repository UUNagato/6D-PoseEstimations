'''
    NOCS decoder renderer
    
    Arguments:
    --obj_folder: The path to the obj model folder
    --data_path: The path to the encoder npz data
    --output_folder: The path to the output images
    --debug: Debug mode, when specified, only 5 images will output
'''
import numpy as np
import glob
import os
import argparse
import cv2          # using cv2 for png load and save might cause a problem of incompatible libpng versions.
import bpy
import queue

parser = argparse.ArgumentParser()
parser.add_argument('--obj_folder', help='The path to the obj model folder', required=True)
parser.add_argument('--data_path', help='The path to the encoder npz data', required=True)
parser.add_argument('--output_folder', help='The path to the output images', required=True)
parser.add_argument('--debug_output_interval', help='Output images every X images', default='50')
parser.add_argument('--debug', help='debug mode, only a few samples will be executed', action='store_true')
parser.add_argument('--remove_material', help='Remove object materials when rendering', action='store_true')

args, _ = parser.parse_known_args()

obj_buf = {}
obj_key_queue = queue.Queue()

obj_folder = args.obj_folder
data_path = args.data_path
output_folder = args.output_folder
remove_material = args.remove_material
output_interval = int(args.debug_output_interval)
debug = args.debug

cropped_size = (128,128)
global_obj_scale = 1.0
near_plane = 0.1
far_plane = 100

def load_obj(obj_path, rot, scale = np.array([global_obj_scale] * 3, dtype=np.float32)):
    # if the list is too long, remove some, obj_buf is FIFO
    print ("len and keys {}, {}".format(len(obj_buf), obj_buf.keys()))
    if len(obj_buf) >= 20:
        # remove 5
        #for i in range(5):     don't know why this method will cause segfault when render.
        #    key = obj_key_queue.get()
        #    obs = obj_buf[key]
        #    for obj in obs:
        #        obj.select = True
        #        bpy.ops.object.delete(use_global=False)
        #    del obj_buf[key]
        empty_obj_buf()
    # find if it's not cached, load it
    if obj_path not in obj_buf:
        print ("Load obj {} into buffer".format(obj_path))
        bpy.ops.import_scene.obj(filepath=os.path.join(obj_folder, obj_path, 'model.obj'))
        obs = bpy.context.selected_editable_objects[:]
        obj_buf[obj_path] = obs
        obj_key_queue.put(obj_path)

    obs = obj_buf[obj_path]
    for obj in obs:
        world_m = np.eye(4)
        world_m[:3,:3] = np.dot(rot, np.diag(scale))
        world_m[:3,3] = np.array([0,0,6],dtype=np.float32)
        obj.matrix_world = world_m.transpose() # numpy matrix need to transpose
    
    # hide all other objects
    for k, obs in obj_buf.items():
        for obj in obs:
            obj.hide = False if k == obj_path else True
            obj.hide_render = False if k == obj_path else True
    

def empty_obj_buf():
    for obs in obj_buf.values():
        for obj in obs:
            obj.select = True
            bpy.ops.object.delete(use_global=False)
    obj_buf.clear()
    while not obj_key_queue.empty():
        obj_key_queue.get()

def get_canonical_config(is_shapenet):
    dict={}
    dict['pad_factor'] = 1.2
    dict['out_shape']=(128,128,3)
    dict['dist_z'] = 6 if is_shapenet else 700
    dict['canonic_lamp_loc'] = np.array([4., 4., 4.]).reshape((3,)) if is_shapenet else np.array([400., 400., 400. - dict['dist_z']]).reshape((3,))
    dict['canonic_diffuse_energy'] = 0.8
    dict['canonic_specular_energy'] = 0.3
    dict['canonic_environment_energy'] = 0.4
    return dict

def set_camera(cam_K,render_dim,scene,cam,clip_start,clip_end):
    from mathutils import Matrix, Vector
    render_w,render_h=render_dim
    fx, fy, ux, uy = cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2]
    scene.render.resolution_x = render_w
    scene.render.resolution_y = render_h
    scene.render.resolution_percentage = 100
    scene.render.alpha_mode = 'TRANSPARENT'

    cam.select = True
    cam.animation_data_clear()

    cam.data.shift_x = (0.5 * render_w - ux) / render_w  # -(ux/w-0.5)
    cam.data.shift_y = (uy - 0.5 * render_h) / render_w

    cam.data.lens = fx / render_w * cam.data.sensor_width
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = fy / fx

    #cam.data.clip_end = clip_end #10 if is_shapenet else 10000
    #cam.data.clip_start = clip_start #0.1 if is_shapenet else 10
    cam.matrix_world = Matrix(((1., 0., 0., 0.), (0., -1., 0., 0.), (0., 0., -1., 0.), (0., 0., 0., 1.)))
    cam.rotation_mode = 'QUATERNION'

def initialize_scene():
    print ("ini:{}".format(bpy.context.scene.objects.keys()))
    # initialize the scene, camera and lights
    # remove cube
    if 'Cube' in bpy.data.objects:
        bpy.data.objects['Cube'].select = True
        bpy.ops.object.delete()
    
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    # Add passes for additionally dumping albedo and normals.
    #bpy.context.scene.render.layers["RenderLayer"].use_pass_normal = True
    #bpy.context.scene.render.layers["RenderLayer"].use_pass_color = True
    bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
    bpy.context.scene.render.image_settings.color_depth = '16'
    bpy.context.scene.render.image_settings.use_zbuffer = True

    # Clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)
    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')

    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    links.new(render_layers.outputs['Z'], depth_file_output.inputs[0])

    # bpy.ops.object.select_all(action='TOGGLE')
    if 'Lamp' in list(bpy.data.objects.keys()):
        bpy.data.objects['Lamp'].select = True  # remove default light
        bpy.ops.object.delete()
    
    canonical_info = get_canonical_config(False)

    bpy.context.scene.world.light_settings.use_environment_light = True
    bpy.context.scene.world.light_settings.environment_energy = canonical_info['canonic_environment_energy']
    bpy.context.scene.world.light_settings.environment_color = 'PLAIN'

    # set point lights
    canonic_lamp_loc = canonical_info['canonic_lamp_loc']
    canonic_diffuse_energy = canonical_info['canonic_diffuse_energy']
    canonic_specular_energy = canonical_info['canonic_specular_energy']

    bpy.ops.object.lamp_add(type='POINT', view_align=False, location=canonic_lamp_loc)
    bpy.ops.object.lamp_add(type='POINT', view_align=False, location=canonic_lamp_loc)

    light_diffuse = bpy.data.objects['Point']
    light_diffuse.data.energy = canonic_diffuse_energy  # np.random.uniform(0.7,0.9)#np.random.normal(1, 0.2)
    light_diffuse.data.use_diffuse = True
    light_diffuse.data.use_specular = False

    light_specular = bpy.data.objects['Point.001']
    light_specular.data.energy = canonic_specular_energy  # np.random.uniform(0.2,0.4)#np.random.normal(1, 0.2)
    light_specular.data.use_diffuse = False
    light_specular.data.use_specular = True

    cur_lamp_loc = canonical_info['canonic_lamp_loc']
    light_diffuse.location=cur_lamp_loc
    light_specular.location=cur_lamp_loc

    scene = bpy.context.scene
    print (scene.objects.keys())
    cam = scene.objects['Camera']
    clip_start,clip_end=near_plane, far_plane

    intrinsics = np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]])
    film_size = (640, 480)
    set_camera(intrinsics, film_size, scene,cam,clip_start,clip_end)

    return render_layers, depth_file_output

def extract_mask_bbox(depth_map):
    '''
        This function returns mask and bbox from depth_map
        return:
            - (y1, x1, y2, x2): bbox
            - mask
    '''
    mask = np.where((depth_map >= near_plane) & (depth_map <= far_plane), 1.0, 0.0)
    horizontal_indices = np.where(np.any(mask, axis=0))[0]
    vertical_indices = np.where(np.any(mask, axis=1))[0]

    mask = 1.0 - mask       # inverse, 1 for background, 0 for foreground

    if horizontal_indices.shape[0]:
        x1, x2 = horizontal_indices[[0, -1]]
        y1, y2 = vertical_indices[[0, -1]]
        x2 += 1
        y2 += 1     # as upper bound, they should not be part of box
    else:
        x1, x2, y1, y2 = 0, 0, 0, 0
    return (y1, x1, y2, x2), mask.astype(np.uint8)

def crop_fixed_padding(img, bbox):
    bbox_center_x = (bbox[3] + bbox[1]) // 2
    bbox_center_y = (bbox[2] + bbox[0]) // 2

    half_size_x = cropped_size[1] // 2
    half_size_y = cropped_size[0] // 2

    start_x = bbox_center_x - half_size_x
    start_y = bbox_center_y - half_size_y

    return img[start_y:start_y+cropped_size[0], start_x:start_x+cropped_size[1]]

def crop_factor_padding(img, bbox, pad_factor=1.2):
    # method 2, multipathaae crop (from Multipath AAE, dataset.py line 208)
    bbox_w = bbox[3] - bbox[1]
    bbox_h = bbox[2] - bbox[0]

    size = int(max(bbox_w, bbox_h) * pad_factor)

    left = int(max(bbox[1]+bbox_w//2 - size // 2, 0))
    right = int(min(left + size, img.shape[1]))
    top = int(max(bbox[0]+bbox_h//2 - size // 2, 0))
    bottom = int(min(top + size, img.shape[0]))

    cropped = img[top:bottom, left:right]
    cropped = cv2.resize(cropped, (cropped_size[1], cropped_size[0]))

    return cropped

def depth_preview(depth):
    copy_depth = depth / 10.0 * 255.0
    #copy_depth -= near_plane
    #copy_depth *= (255.0 / (far_plane - near_plane)
    copy_depth = np.clip(copy_depth, 0.0, 255.0)

    return copy_depth.astype(np.uint8)

def float_2_uint8(img):
    img = np.clip(img * 255, 0.0, 255.0)
    return img.astype(np.uint8)
    
def main():
    print ("Start to process data with configuration:\ndata path:{}\nobj path:{}\noutput path:{}\n \
            cropped_size:{}\ndebug mode:{}".format(data_path, \
        obj_folder, output_folder, cropped_size, debug))

    _, depth_file_output = initialize_scene()
    # find all npz file
    npz_files = glob.glob(os.path.join(data_path, './*.npz')) # ./ is used for Windows
    print (npz_files)

    scene = bpy.context.scene

    for npz_file in npz_files:
        # load it
        print (npz_file)
        with np.load(npz_file) as data:
            empty_obj_buf()
            # empty obj_buf
            if 'obj_list' not in data or 'matrix_rot_y' not in data:
                print ("[Error]npz file {} doesn't have key 'obj_file' or 'matrix_rot_y'".format(npz_file))
                continue
            # read sample number
            rot_matrices = data['matrix_rot_y']
            obj_list = data['obj_list']
            
            if rot_matrices.shape[0] != obj_list.shape[0]:
                print ("[Error]npz file {} has different sample number of 'matrix_rot_y' and 'obj_list'".format(npz_file))
                continue
            
            num_sample = rot_matrices.shape[0] if not debug else 2000
            output_data = {'scene_id':data['scene_id'], 'image_id':data['image_id'], 'class_id':data['class_id'], 'bbox':data['bbox'], \
                            'matrix_rot_y':rot_matrices, 'matrix_tra_y':data['matrix_tra_y'], 'obj_list':obj_list, \
                            'bgr_y_render':np.zeros((num_sample, cropped_size[0], cropped_size[1], 3), dtype=np.uint8), \
                            'bgr_y_render_aae':np.zeros((num_sample, cropped_size[0], cropped_size[1], 3), dtype=np.uint8), \
                            'depth_y':np.zeros((num_sample, cropped_size[0], cropped_size[1]), dtype=np.float16), \
                            'mask_y':np.zeros((num_sample, cropped_size[0], cropped_size[1]), dtype=np.bool) }

            npz_name = os.path.basename(npz_file)
            npz_name = npz_name[:-4]

            for i in range(num_sample):
                # print ("Processing:{}/{}".format(i, num_sample), end='\r')
                load_obj(obj_list[i], rot_matrices[i])
                # print ("Finish loading obj")
                # render, I hope I can avoid this IO but directly access the pixel values.
                # Unfortunately, seems there's not such a method.
                render_img_path = os.path.join(output_folder, '{}_{}_NOCS'.format(npz_name, i))
                scene.render.image_settings.file_format = 'PNG'
                scene.render.filepath = render_img_path
                depth_file_output.base_path=''
                depth_file_output.format.file_format = 'OPEN_EXR'
                depth_file_output.file_slots[0].path = scene.render.filepath + "_depth"
                # print ("Ready to render")
                bpy.ops.render.render(write_still=True)
                
                # post-process with numpy and cv2
                # use blender's native way to load to avoid libpng incompatible error.
                #tmp_img = bpy.data.images.load(render_img_path + '.png')
                #img_size = tmp_img.size + tmp_img.channels
                #color_img = np.array(tmp_img.pixels, dtype=np.float32).reshape(img_size)
                #bpy.data.images.remove(tmp_img)     # release resource
                #color_img = float_2_uint8(color_img)
                # print ("Read images\n")
                color_img = cv2.imread(render_img_path + '.png')
                depth_img = cv2.imread(render_img_path + '_depth0001.exr', cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)[..., 2] # actually, channel 1,2,3 are same.
                
                # print (depth_img[220:240,320:340])
                # compute mask
                #print ("Compute Masks\n")
                bbox, masks = extract_mask_bbox(depth_img)

                # use two methods to crop
                # method 1, fixed crop
                #print ("Crop images 1\n")
                m1_cropped_img = crop_fixed_padding(color_img, bbox)
                m1_cropped_depth = crop_fixed_padding(depth_img, bbox)
                m1_cropped_mask = crop_fixed_padding(masks, bbox)

                # method 2, factor padding
                #print ("Crop images 2\n")
                m2_cropped_img = crop_factor_padding(color_img, bbox)

                # save
                #print ("Save to output data\n")
                output_data['bgr_y_render'][i] = m1_cropped_img
                output_data['bgr_y_render_aae'][i] = m2_cropped_img
                output_data['depth_y'][i] = m1_cropped_depth
                output_data['mask_y'][i] = m1_cropped_mask.astype(np.bool)

                # if it's output interval, save it, the output depth is only for preview because cv2 cannot export float16
                if i % output_interval == 0:
                    cv2.imwrite(render_img_path + "_m1crop_color.png", m1_cropped_img)
                    cv2.imwrite(render_img_path + "_m2crop_color.png", m2_cropped_img)
                    cv2.imwrite(render_img_path + "_crop_depth.png", depth_preview(m1_cropped_depth))
                    cv2.imwrite(render_img_path + "_crop_mask.png", m1_cropped_mask * 255)
                else:
                    # else, delete rendered result
                    os.remove(render_img_path + '.png')
                    os.remove(render_img_path + '_depth0001.exr')
            
            save_path = os.path.splitext(npz_file)[0] + '_decoder.npz'
            print ("saving results to {}".format(save_path))
            #np.savez_compressed(save_path, **output_data)
            print ("saved results to {}".format(save_path))

            if debug:
                break

    empty_obj_buf()

if __name__ == '__main__':
    main()