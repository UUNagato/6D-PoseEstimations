# a blender script to render NOCS ground truth
# Notice line 85 overrides the obj initial 90 degree x rotation
# Line 221 is used to count initial 90 degree to m2c matrix
# So, if you want to count the initial 90 degree as part of m2c matrix. Use line 86 and line 221, else use line 85 and line 220
import os, sys
sys.path.append('./')
import numpy as np
import pickle
import bpy
import math

ground_truth_file = 'D:/MSRA/CAMERADataset/00000/GT00000/results_val_00000_0000.pkl'
dataset_path = 'D:/MSRA/BlenderRenderer/camera_obj_models'

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

def load_obj(path_obj, scale_input, remove_doubles,edge_split):
    # Preprocess object
    bpy.ops.import_scene.obj(filepath=path_obj)
    obs = bpy.context.selected_editable_objects[:]

    for obj in obs:
        # here I override the matrix_world with scale only because the model will have 90 degree x rotaion when imported.
        obj.matrix_world = np.diag(np.append(scale_input, 1.0))
        #obj.scale = scale_input               # use this if you want to count initial 90 degree into m2c matrix.
        # apply transformation

        #if remove_doubles:
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

# decompose transform matrix and return translation, scale, rotation
def decompose_transform_matrix(matrix):
    assert (matrix.shape == (4, 4))
    # get translation first
    translation = matrix[0:3, 3]
    # get scale & rotation
    scale = np.array([1.0, 1.0, 1.0])
    rotation = np.ones((3, 3), dtype = matrix.dtype)
    for i in range(3):
        norm = np.linalg.norm(matrix[:, i])
        scale[i] = norm
        rotation[0:3,i] = matrix[0:3,i] / norm
    
    return translation, scale, rotation

def quaternion_from_matrix(matrix, isprecise=False):
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q

def get_q_c2m_gl(R_c2m):
    R_aug_c2m_gl = np.eye(4)
    R_aug_c2m_gl[:3, :3] =R_c2m

    R_aug_c2m_gl[:, 1] = R_aug_c2m_gl[:, 1]
    R_aug_c2m_gl[:, 2] = R_aug_c2m_gl[:, 2]
    q_c2m_gl=quaternion_from_matrix(R_aug_c2m_gl)
    return q_c2m_gl

# read in gt
instance_num = 0
with open(ground_truth_file, 'rb') as f:
    data = pickle.load(f)
    instance_num = data['gt_RTs'].shape[0]

if instance_num == 0:
    print('no instance detected.')
    exit()

# use blender

intrinsics = np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]])
film_size = (640, 480)

canonical_info=get_canonical_config(False)
init_scene()

# remove default obj
if 'Cube' in bpy.data.objects:
    bpy.data.objects['Cube'].select = True
    bpy.ops.object.delete()

obj_file = data['obj_list'][0]
# fullpath
obj_fileparts = obj_file.split('/')
obj_filepath = os.path.join(dataset_path, obj_fileparts[-4], obj_fileparts[-3], obj_fileparts[-2], obj_fileparts[-1])

# m2c is not only the gt_RTs, but also a camera rotation is applied. This is because the camera is towards negative z instead of positive z which we expected.
obj_transform = data['gt_RTs'][0]
#obj_transform = np.dot(data['gt_RTs'][0], np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]]))
m2c_matrix = np.dot(np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]),obj_transform)
t_m2c, scale, R_m2c = decompose_transform_matrix(m2c_matrix)
load_obj(obj_filepath, scale, True, True)

light_diffuse,light_specular=init_lighting(canonical_info)
##############################################
# Set camera intrinsic
scene = bpy.context.scene
cam = scene.objects['Camera']
clip_start,clip_end=10,10000

set_camera(intrinsics, film_size, scene,cam,clip_start,clip_end)

R_c2m =  np.linalg.inv(R_m2c.copy())
t_c2m = R_c2m[:3,:3].dot(t_m2c * -1.0)

#camera_matrix = np.zeros([4, 4], dtype=np.float32)
#camera_matrix[0:3, 0:3] = R_c2m
#camera_matrix[0:3, 3] = t_c2m
#camera_matrix[3,:] = np.array([0, 0, 0, 1])

q_c2m_gl=get_q_c2m_gl(R_c2m)

cam.rotation_quaternion=q_c2m_gl
cam.location= t_c2m
#cam.matrix_world = camera_matrix.transpose()

#Process lighting
cur_lamp_loc = canonical_info['canonic_lamp_loc']
light_diffuse.location=cur_lamp_loc
light_specular.location=cur_lamp_loc

scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = os.path.join('D:/MSRA/BlenderRenderer', 'test_render_NOCS')

bpy.ops.render.render(write_still=True)