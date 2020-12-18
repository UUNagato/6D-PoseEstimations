'''
    This script is used to render REAL dataset. It uses cubes as objects (because there's no corresponding objects in GT files)
    This script is used mainly to validate the meaning of RTs and verify the process of converting to camera space.
'''

# a blender script to render NOCS ground truth
import os, sys
sys.path.append('./')
import numpy as np
import pickle
import bpy

ground_truth_file = '/home/v-xli2/NOCS_CVPR2019/output/real_test_20201104T1054/results_test_scene_1_0000.pkl'
# dataset_path = 'D:/MSRA/BlenderRenderer/camera_obj_models'

def init_scene():
    import bpy
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    # Add passes for additionally dumping albedo and normals.
    #bpy.context.scene.render.layers["RenderLayer"].use_pass_normal = True
    #bpy.context.scene.render.layers["RenderLayer"].use_pass_color = True
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_depth = '8'  # args.color_depth
    bpy.context.scene.render.image_settings.use_zbuffer = True

    # Clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)
    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')

def init_lighting(canonical_info):
    import bpy
    ############################################
    # Set Lighting
    bpy.ops.object.select_all(action='TOGGLE')
    if 'Lamp' in list(bpy.data.objects.keys()):
        bpy.data.objects['Lamp'].select = True  # remove default light
    bpy.ops.object.delete()

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

    return light_diffuse,light_specular


def get_canonical_config(is_shapenet):
    dict={}
    dict['pad_factor'] = 1.2
    dict['out_shape']=(128,128,3)
    dict['dist_z'] = 4 if is_shapenet else 700
    dict['canonic_lamp_loc'] = np.array([4., 4., 4.]).reshape((3,)) if is_shapenet else np.array([400., 400., 400. - dict['dist_z']]).reshape((3,))
    dict['canonic_diffuse_energy'] = 0.8
    dict['canonic_specular_energy'] = 0.3
    dict['canonic_environment_energy'] = 0.4
    return dict

def load_obj(path_obj,transform_matrix,remove_doubles,edge_split):
    import mathutils
    # Preprocess object
    bpy.ops.import_scene.obj(filepath=path_obj)
    obs = bpy.context.selected_editable_objects[:]
    for ob in obs:
        # apply transformation
        transform_matrix[2, :] *= -1    # flip z
        ob.matrix_world = transform_matrix
        print (ob.matrix_world)        #if remove_doubles:
        #    ob.mode_set(mode='EDIT')
        #    bpy.ops.mesh.remove_doubles()
        #    ob.mode_set(mode='OBJECT')
        #if edge_split:
        #    bpy.ops.object.modifier_add(type='EDGE_SPLIT')
        #    bpy.context.object.modifiers["EdgeSplit"].split_angle = 1.32645
         #   bpy.ops.object.modifier_apply(apply_as='DATA', modifier="EdgeSplit")

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

    #cam.data.clip_end = clip_end#10 if is_shapenet else 10000
    #cam.data.clip_start = clip_start#0.1 if is_shapenet else 10
    cam.matrix_world = Matrix(((1., 0., 0., 0.), (0., -1., 0., 0.), (0., 0., -1., 0.), (0., 0., 0., 1.)))
    cam.rotation_mode = 'QUATERNION'

# read in gt
instance_num = 0
with open(ground_truth_file, 'rb') as f:
    data = pickle.load(f)
    instance_num = data['gt_RTs'].shape[0]

if instance_num == 0:
    print('no instance detected.')
    exit()

# use blender

intrinsics = np.array([[591.0125, 0, 322.525], [0., 590.16775, 244.11084], [0., 0., 1.]])
film_size = (640, 480)

canonical_info=get_canonical_config(False)
init_scene()

# remove default obj
if 'Cube' in bpy.data.objects:
    bpy.data.objects['Cube'].select = True
    bpy.ops.object.delete()

for i in range(instance_num):
    #obj_file = data['obj_list'][i]
    # fullpath
    #obj_fileparts = obj_file.split('/')
    #obj_filepath = os.path.join(dataset_path, obj_fileparts[-4], obj_fileparts[-3], obj_fileparts[-2], obj_fileparts[-1])
    #load_obj(obj_filepath, data['gt_RTs'][i].transpose(), True, True)
    # create a cube
    bpy.ops.mesh.primitive_cube_add()
    gt_scale = data['pred_scales'][i]
    gt_RT = data['pred_RTs'][i]
    world_mat = gt_RT.copy()
    for c in range(3):
        world_mat[:,c] = gt_RT[:,c] * gt_scale[c] * 0.5
    mat = bpy.data.materials.new("TMP")
    mat.diffuse_color = np.random.rand(3)
    bpy.context.object.matrix_world = world_mat.transpose()
    bpy.context.object.active_material = mat

light_diffuse,light_specular=init_lighting(canonical_info)
##############################################
# Set camera intrinsic
scene = bpy.context.scene
cam = scene.objects['Camera']
clip_start,clip_end=10,10000

set_camera(intrinsics, film_size, scene,cam,clip_start,clip_end)



#Process lighting
cur_lamp_loc = canonical_info['canonic_lamp_loc']
light_diffuse.location=cur_lamp_loc
light_specular.location=cur_lamp_loc

scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = os.path.join('/home/v-xli2/blender-2.78c/NOCSRender/', 'test_render_NOCS')

bpy.ops.render.render(write_still=True)