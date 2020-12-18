# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.
#
# Example:
# /home/yuexin/blender-2.78c-linux-glibc219-x86_64/blender --background --python render_blender.py -- --views 10 ./obj_11.py
#

import argparse, sys, os
import numpy as np
import math
sys.path.append('./')
from pysixd_stuff.pysixd import transform
from pysixd_stuff.pysixd import view_sampler
import data_utils
import bpy
from mathutils import Matrix, Vector
import cv2

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('obj', type=str,
                    help='Path to the obj file to be rendered.')
parser.add_argument('--sid',type=int)
parser.add_argument('--eid',type=int)

#parser.add_argument('--remove_doubles', type=bool, default=False,
#                    help='Remove double vertices to improve mesh quality.')
#parser.add_argument('--edge_split', type=bool, default=False,
#                    help='Adds edge split filter.')
#parser.add_argument('--depth_scale', type=float, default=1.4,
#                    help='Scaling that is applied to depth. Depends on size of mesh. Try out various values until you get a good result. Ignored if format is OPEN_EXR.')
#parser.add_argument('--color_depth', type=str, default='8',
#                   help='Number of bit per channel used for output. Either 8 or 16.')
#parser.add_argument('--format', type=str, default='PNG',help='Format of files generated. Either PNG or OPEN_EXR')


argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

path_obj= args.obj
output_dir='D:/MSRA/BlenderRenderer/embedding92232s_camera/'
is_shapenet= path_obj.split('.')[-1]=='obj'
cam_K=np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]]).reshape((3,3))
#cam_K=np.array([1075.65, 0, 720 / 2, 0, 1073.90, 540 / 2, 0, 0, 1]).reshape(3, 3)
canonical_info=data_utils.get_canonical_config(is_shapenet)
render_w,render_h= 640,480
out_shape = canonical_info['out_shape']
pad_factor = canonical_info['pad_factor']
dist_z = canonical_info['dist_z']

path_splits = path_obj.split('/')
model_identifier = path_splits[-3] + '_' + path_splits[-2] if is_shapenet else '11'
dir_imgs = os.path.join(os.path.join(output_dir, model_identifier), 'imgs')
path_obj_bbs=os.path.join(os.path.join(output_dir, model_identifier), 'obj_bbs.npy')
path_rot=os.path.join(os.path.join(output_dir, model_identifier), 'rot_infos.npz')
depth_scale=1.

#############################################
data_utils.init_scene()
data_utils.load_obj(path_obj,depth_scale=depth_scale,remove_doubles=True,edge_split=True)
light_diffuse,light_specular=data_utils.init_lighting(canonical_info)

##############################################
# Set camera intrinsic
scene = bpy.context.scene
cam = scene.objects['Camera']
if not is_shapenet:
    clip_start,clip_end=10,10000

elif model_identifier.split('_')[0] in ['03642806']:
    clip_start,clip_end=dist_z-1.2, dist_z+1.2
else:
    clip_start,clip_end=0.1,100
data_utils.set_camera(cam_K,(render_w,render_h),scene,cam,clip_start,clip_end)
#double check camera
if False:
    fx,fy,ux,uy=cam_K[0,0],cam_K[1,1],cam_K[0,2],cam_K[1,2]
    print(fx,fy,ux,uy)
    fx2=cam.data.lens/cam.data.sensor_width*scene.render.resolution_x
    fy2=fx2*scene.render.pixel_aspect_y/scene.render.pixel_aspect_x

    cx2=scene.render.resolution_x*(0.5-cam.data.shift_x)
    cy2=scene.render.resolution_y*0.5+scene.render.resolution_x*cam.data.shift_y
    print(fx2,fy2,cx2,cy2,'\n',scene.render.pixel_aspect_y,cam.data.sensor_height,scene.render.pixel_aspect_x,cam.data.sensor_width)


view_Rs = data_utils.viewsphere_for_embedding_v2(num_sample_views=2000, num_cyclo=36, use_hinter=True)
embedding_size=view_Rs.shape[0]
if args.sid != 0:
    obj_bbs = np.load(path_obj_bbs)
else:
    obj_bbs = np.empty((embedding_size, 4))

debug=False
if debug:
    img_poses=np.zeros((128,128*7,3),dtype=np.uint8)

for iid in range(args.sid, min(embedding_size,args.eid)):
    R=view_Rs[iid]
    #R_m2c= np.array([-0.05081750, 0.99733198, -0.05241400, 0.93482399, 0.02903240, -0.35392201, -0.35145599, -0.06698330, -0.93380499]).reshape((3, 3))
    #t_m2c=np.array([50.10567277, -77.01673570, 958.81105256]).reshape((3,1))
    #R_m2c = np.array([0.22732917, -0.13885391, -0.96386789, -0.93565258, 0.24323566, -0.25571487, 0.26995383, 0.95997719, -0.07462456]).reshape((3, 3))
    #t_m2c = np.array([-42.09236434, -97.50670441, 799.79210049]).reshape((3, 1))

    R_m2c=R.copy()
    t_m2c=np.array([0,0,dist_z]).reshape((3,1))

    R_c2m =  np.linalg.inv(R_m2c.copy())
    t_c2m = -R_c2m[:3,:3].dot(t_m2c)
    q_c2m_gl=data_utils.get_q_c2m_gl(R_c2m)

    #R_c2m*(R_m2c M+T_m2c)+T_c2m=M
    cam.location= t_c2m
    cam.rotation_quaternion=q_c2m_gl
    print(cam.location)
    print(cam.matrix_world)

    #Process lighting
    cur_lamp_loc = -R_c2m[:3,:3].dot(canonical_info['canonic_lamp_loc'])
    light_diffuse.location=cur_lamp_loc
    light_specular.location=cur_lamp_loc

    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = os.path.join(dir_imgs, './{:05d}'.format(iid))

    bpy.ops.render.render(write_still=True)

    #####Post Process####
    print("Post process path:%s" % os.path.join(dir_imgs, './{:05d}.png'.format(iid)))
    img_bgra = cv2.imread(os.path.join(dir_imgs, './{:05d}.png'.format(iid)), cv2.IMREAD_UNCHANGED)
    depth_y=img_bgra[:,:,3]
    for cc in range(0,3):
        img_bgra[:,:,cc]=np.where(depth_y,img_bgra[:,:,cc],255)

    if False:
        cv2.imwrite(os.path.join(output_dir,'./{:s}_{:d}.png'.format(model_identifier,iid)),img_bgra[:,:,:3].copy())

    ys, xs = np.nonzero(depth_y > 0)
    obj_bb = view_sampler.calc_2d_bbox(xs, ys, (render_w,render_h))
    obj_bbs[iid]=obj_bb

    bgr_y = img_bgra[:,:,0:3].copy()
    resized_bgr_y = data_utils.extract_square_patch(bgr_y, obj_bb, pad_factor, resize=out_shape[:2],
                                                    interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(os.path.join(dir_imgs, './{:05d}.png'.format(iid)),resized_bgr_y)

    if debug:
        img_poses[:,iid*128:(iid+1)*128,:]=resized_bgr_y.copy()
    if not is_shapenet:# and debug:
        obj_bbs0=np.load('../Edge-Network/embedding20s/11/obj_bbs.npy')
        print(obj_bbs0[iid],obj_bb)

if debug:
    cv2.imwrite(os.path.join(output_dir,'./{:s}.png'.format(model_identifier)),img_poses)
else:
    np.save(path_obj_bbs,obj_bbs)
    np.savez(path_rot, rots=view_Rs)






'''
if data in ['real_train', 'real_test']:
    intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
else: ## CAMERA data
    intrinsics = np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]])
    
gt_pkl_path = os.path.join(gt_dir, 'results_{}_{}_{}.pkl'.format(data, image_path_parsing[-2], image_path_parsing[-1]))
print(gt_pkl_path)
if (os.path.exists(gt_pkl_path)):
    with open(gt_pkl_path, 'rb') as f:
        gt = cPickle.load(f)
    result['gt_RTs'] = gt['gt_RTs']
    if 'handle_visibility' in gt:
        result['gt_handle_visibility'] = gt['handle_visibility']
        assert len(gt['handle_visibility']) == len(gt_class_ids)
        print('got handle visibiity.')
    else: 
        result['gt_handle_visibility'] = np.ones_like(gt_class_ids)
'''
