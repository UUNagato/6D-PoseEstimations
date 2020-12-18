# -*- coding: utf-8 -*-
# This is to sample rotations to train the Autoencoder.
# For each sampled rotation, its BGR image(foreground only) for Decoder GT and Encoder input, the foreground masks, and the rotation of decoder GT is generated and preserved.

import numpy as np
import cv2
import os,sys
sys.path.append('./')
import data_utils
from pysixd_stuff.pysixd import inout,transform,view_sampler


verbose=True
class Model(object):
    def __init__(self, dir_dataset,path_model, saved_file_name, num_per_batch):
        self.shape_c3 = (128,128,3)
        self.num_per_batch= num_per_batch
        self.dir_dataset=dir_dataset#path to save rendered training data
        self.path_model=path_model#path of mesh
        self.saved_file_name=saved_file_name

        self.bgr_x = np.empty((self.num_per_batch,) + self.shape_c3, dtype=np.uint8)
        self.bgr_y = np.empty((self.num_per_batch,) + self.shape_c3, dtype=np.uint8)

        self.mask_x = np.empty((self.num_per_batch,) + self.shape_c3[:2], dtype=bool)
        self.mask_y = np.empty((self.num_per_batch,)+self.shape_c3[:2],dtype=bool)

        self.matrix_rot_y=np.empty((self.num_per_batch,)+(3,3),dtype=np.float32)


    def combine_rendered_batches(self,num_batches):
        num_total=self.num_per_batch*num_batches
        self.bgr_x = np.empty((num_total,) + self.shape_c3, dtype=np.uint8)
        self.bgr_y = np.empty((num_total,) + self.shape_c3, dtype=np.uint8)

        self.mask_x = np.empty((num_total,) + self.shape_c3[:2], dtype=bool)
        self.mask_y = np.empty((num_total,)+self.shape_c3[:2],dtype=bool)

        self.matrix_rot_y=np.empty((num_total,)+(3,3),dtype=np.float32)
        self.noof_obj_pixels = np.empty((num_total,), dtype=np.uint8)
        print('Size',self.bgr_x.shape)

        for i in range(0,num_batches):
            current_file_name=os.path.join(self.dir_dataset, self.saved_file_name + '{0}.npz'.format(i))
            training_data = np.load(current_file_name)
            self.bgr_x[i*self.num_per_batch:(i+1)*self.num_per_batch]=training_data['bgr_x'].astype(np.uint8)
            self.bgr_y[i*self.num_per_batch:(i+1)*self.num_per_batch]=training_data['bgr_y'].astype(np.uint8)

            self.mask_y[i*self.num_per_batch:(i+1)*self.num_per_batch] = training_data['mask_y'].astype(bool)
            self.mask_x[i*self.num_per_batch:(i+1)*self.num_per_batch] = training_data['mask_x'].astype(bool)
            self.matrix_rot_y[i*self.num_per_batch:(i+1)*self.num_per_batch]=training_data['matrix_rot_y'].astype(np.float32)

            if verbose:
                vis_mask_x = np.where(self.mask_x[(i+1)*self.num_per_batch-1],0,255).astype(np.uint8)
                vis_mask_y = np.where(self.mask_y[(i+1)*self.num_per_batch-1],0,255).astype(np.uint8)
                cv2.imshow('bgr_x', self.bgr_x[(i+1)*self.num_per_batch-1])
                cv2.imshow('mask_x', vis_mask_x)
                cv2.imshow('mask_y', vis_mask_y)
                cv2.imshow('bgr_y', self.bgr_y[(i+1)*self.num_per_batch-1])
                cv2.waitKey()
                print('rot_y:',self.matrix_rot_y[(i+1)*self.num_per_batch-1])
        current_file_name = os.path.join(self.dir_dataset, self.saved_file_name + '.npz')
        np.savez(current_file_name, bgr_x=self.bgr_x,bgr_y=self.bgr_y, mask_x=self.mask_x, mask_y=self.mask_y, matrix_rot_y=self.matrix_rot_y)

    def load_training_images(self):
        current_file_name = os.path.join(self.dir_dataset, self.saved_file_name + '.npz')
        training_data = np.load(current_file_name)
        self.bgr_x = training_data['bgr_x'].astype(np.uint8)
        self.mask_x = training_data['mask_x']

        self.bgr_y = training_data['bgr_y'].astype(np.uint8)
        self.mask_y = training_data['mask_y']
        self.matrix_rot_y=training_data['matrix_rot_y'].astype(np.float32)

        print('Size',self.bgr_x.shape)

        if verbose:
            vis_mask_x = np.where(self.mask_x[-1],0,255).astype(np.uint8)
            vis_mask_y = np.where(self.mask_y[-1],0,255).astype(np.uint8)

            cv2.imshow('bgr_x',self.bgr_x[-1])
            cv2.imshow('mask_x',vis_mask_x)
            cv2.imshow('mask_y',vis_mask_y)
            cv2.imshow('bgr_y',(self.bgr_y[-1]).astype(np.uint8))
            cv2.waitKey()
            print('check rot: ',self.matrix_rot_y[-1])

    def render_batch_training_images(self,render_dims,cam_K,batch_id,depth_scale=1.,texture_img=None):
        from pysixd_stuff.pysixd import renderer_vt
        current_file_name = os.path.join(self.dir_dataset, self.saved_file_name + '{0}.npz'.format(batch_id))
        H, W = self.shape_c3[0],self.shape_c3[1]
        K = np.array(cam_K).reshape(3, 3)
        clip_near = float(10)
        clip_far = float(10000)
        pad_factor = float(1.2)
        crop_offset_sigma = float(20)
        t = np.array([0, 0, float(700)])#for obj 01, #float(700)for obj 2-4

        import progressbar
        bar = progressbar.ProgressBar(
            maxval=self.num_per_batch,
            widgets=[' [', progressbar.Timer(), ' | ',
                     progressbar.Counter('%0{}d / {}'.format(len(str(self.num_per_batch)), self.num_per_batch)), ' ] ', progressbar.Bar(), ' (',
                     progressbar.ETA(), ') ']
        )
        bar.start()

        model = inout.load_ply(self.path_model)

        model['pts']*=depth_scale

        for i in np.arange(self.num_per_batch):
            bar.update(i)

            R = transform.random_rotation_matrix()[:3, :3]
            im_size = (render_dims[0], render_dims[1])
            rgb_x, depth_x = renderer_vt.render_phong(model, im_size, K.copy(), R, t, clip_near=clip_near,
                                                  clip_far=clip_far,texture=texture_img, mode='rgb+depth', random_light=True)
            rgb_y, depth_y = renderer_vt.render_phong(model, im_size, K.copy(), R, t, clip_near=clip_near,
                                                  clip_far=clip_far,texture=texture_img, mode='rgb+depth', random_light=False)


            #cv2.imshow('rgbx',rgb_x)
            #cv2.imshow('rgby',rgb_y)
            #cv2.waitKey()

            bgr_x = rgb_x.copy()
            bgr_y = rgb_y.copy()
            for cc in range(0, 3):
                bgr_x[:,:,cc]=rgb_x[:,:,2-cc]
                bgr_y[:,:,cc]=rgb_y[:,:,2-cc]

            ys, xs = np.nonzero(depth_x > 0)
            try:
                obj_bb = view_sampler.calc_2d_bbox(xs, ys, render_dims)
            except ValueError as e:
                print('Object in Rendering not visible. Have you scaled the vertices to mm?')
                break

            x, y, w, h = obj_bb

            rand_trans_x = np.random.uniform(-crop_offset_sigma, crop_offset_sigma)
            rand_trans_y = np.random.uniform(-crop_offset_sigma, crop_offset_sigma)

            size = int(np.maximum(h, w) * pad_factor)
            left = int(x + w / 2 - size / 2 + rand_trans_x)
            right = int(x + w / 2 + size / 2 + rand_trans_x)
            top = int(y + h / 2 - size / 2 + rand_trans_y)
            bottom = int(y + h / 2 + size / 2 + rand_trans_y)


            bgr_x = bgr_x[top:bottom, left:right]
            depth_x = depth_x[top:bottom, left:right]
            bgr_x = cv2.resize(bgr_x, (W, H), interpolation=cv2.INTER_NEAREST)
            depth_x = cv2.resize(depth_x, (W, H), interpolation=cv2.INTER_NEAREST)

            mask_x = depth_x == 0.

            ys, xs = np.nonzero(depth_y > 0)
            obj_bb = view_sampler.calc_2d_bbox(xs, ys, render_dims)

            bgr_y = data_utils.extract_square_patch(bgr_y, obj_bb, pad_factor, resize=(W, H), interpolation=cv2.INTER_NEAREST)
            depth_y = data_utils.extract_square_patch(depth_y, obj_bb, pad_factor, resize=(W, H), interpolation=cv2.INTER_NEAREST)
            mask_y = depth_y==0


            self.bgr_x[i] = bgr_x
            self.mask_x[i] = mask_x
            self.bgr_y[i]= bgr_y
            self.mask_y[i] = mask_y
            self.matrix_rot_y[i]=R


            if i%100==0:
                path_out_dir=os.path.join(self.dir_dataset,'imgs')
                if not os.path.exists(path_out_dir):
                    os.makedirs(path_out_dir)
                cv2.imwrite(os.path.join(path_out_dir,'{0}_{1}_x_bgr.png'.format(batch_id,i)),self.bgr_x[i])
                cv2.imwrite(os.path.join(path_out_dir,'{0}_{1}_y_bgr.png'.format(batch_id,i)),self.bgr_y[i])
        bar.finish()
        np.savez(current_file_name, bgr_x=self.bgr_x, mask_x=self.mask_x, bgr_y=self.bgr_y, mask_y=self.mask_y, matrix_rot_y=self.matrix_rot_y)


    def render_batch_training_images_blender(self,render_dims,cam_K,batch_id,depth_scale=1.,is_shapenet=False,debug=False):
        import bpy
        if debug:
            cam_K = np.array([1075.65091572, 0.0, 373.06888344, 0.0, 1073.90347929, 257.72159802, 0.0, 0.0, 1.0]).reshape(3, 3)
            cam_K = np.array([1075.65091572, 0.0, 375.06888344, 0.0, 1073.90347929, 306.72159802, 0.0, 0.0, 1.0]).reshape(3,3)
            render_dims=(720,540)

        render_w, render_h = render_dims  # 640,480
        canonical_info = data_utils.get_canonical_config(is_shapenet)
        pad_factor = canonical_info['pad_factor']
        dist_z = canonical_info['dist_z']

        path_splits= self.path_model.split('/')
        model_identifier = path_splits[-3]+'_' + path_splits[-2] if is_shapenet else '11'
        current_file_name = os.path.join(self.dir_dataset,self.saved_file_name + '{0}.npz'.format(batch_id))

        data_utils.init_scene()
        data_utils.load_obj(self.path_model, depth_scale=depth_scale, remove_doubles=False, edge_split=False)
        light_diffuse, light_specular = data_utils.init_lighting(canonical_info)

        # Set camera intrinsic
        scene = bpy.context.scene
        cam = scene.objects['Camera']
        if not is_shapenet:
            clip_start, clip_end = 10, 10000

        elif model_identifier.split('_')[0] in ['03642806']:
            clip_start, clip_end = dist_z - 1.2, dist_z + 1.2
        else:
            clip_start, clip_end = 0.1, 100
        data_utils.set_camera(cam_K, (render_w, render_h), scene, cam, clip_start, clip_end)

        #view_Rs = data_utils.viewsphere_for_embedding_v2(num_sample_views=400, num_cyclo=20, use_hinter=False)
        for i in np.arange(self.num_per_batch):
            R_m2c = transform.random_rotation_matrix()[:3, :3]
            t_m2c = np.array([0, 0, dist_z]).reshape((3, 1))

            if debug:
                '''
                Rt=np.array([[ 0.40075716, -0.00969288,  0.07055298,  0.30217341],
                            [-0.05753401, -0.28171718,  0.28810284, -0.19431935],
                            [ 0.04197039, -0.29363149, -0.27874184,  0.97399586],
                            [ 0.        ,  0.        ,  0.        ,  1.        ]]).reshape((4,4))

                Rt=np.array([[-0.09787803, -0.01090088,  0.44777024, -0.37836784],
                        [-0.32151401, -0.31739694, -0.07800668, -0.34349534],
                        [ 0.3118425 , -0.33066216,  0.0601157 ,  1.11896169],
                        [ 0.        ,  0.        ,  0.        ,  1.        ]]).reshape((4,4))


                scale=[0.52288 , 0.442578, 0.728506]
                #for cc in range(0,3):
                #    Rt[cc,:]/=scale[cc]

                R_m2c=Rt[:3,:3]
                R_m2c=R_m2c/np.linalg.norm(R_m2c,ord=2)
                R_m2c=np.linalg.inv(R_m2c)
                t_m2c=np.array([0,0,3]).reshape((3,))#Rt[:3,3]
                '''

                R_m2c = np.array([0.22732917, -0.13885391, -0.96386789, -0.93565258, 0.24323566, -0.25571487, 0.26995383, 0.95997719, -0.07462456]).reshape((3, 3))
                t_m2c = np.array([-42.09236434, -97.50670441, 799.79210049]).reshape((3, 1))

                R_m2c=np.array([0.10141005, -0.99483831, 0.00350530, 0.51664706, 0.05567581, 0.85438647, -0.85017138, -0.08483250, 0.51962589]).reshape((3,3))
                t_m2c=np.array([-88.87238397, -58.73935977, 823.08771695]).reshape((3,1))
                #obj_bb: [229, 184, 62, 90]]

            R_c2m = np.linalg.inv(R_m2c.copy())
            t_c2m = -R_c2m[:3, :3].dot(t_m2c)

            q_c2m_gl = data_utils.get_q_c2m_gl(R_c2m)

            # R_c2m*(R_m2c M+T_m2c)+T_c2m=M
            cam.location = t_c2m
            cam.rotation_quaternion = q_c2m_gl
            print(cam.location)
            print(cam.matrix_world)

            #render canonic
            cur_lamp_loc = -R_c2m[:3, :3].dot(canonical_info['canonic_lamp_loc'])
            light_diffuse.location = cur_lamp_loc
            light_specular.location = cur_lamp_loc
            light_diffuse.data.energy = canonical_info['canonic_diffuse_energy']
            light_specular.data.energy = canonical_info['canonic_specular_energy']

            scene.render.image_settings.file_format = 'PNG'  # set output format to .png
            scene.render.filepath =  os.path.join(os.path.join(self.dir_dataset, model_identifier), 'temp_y.png'.format(i))
            bpy.ops.render.render(write_still=True)  # render still

            #render augmented
            random_lamp_loc = (10.*np.random.random(3)) if is_shapenet else 1000. * np.random.random(3)
            if not is_shapenet:
                random_lamp_loc[2]-=dist_z
            cur_lamp_loc = -R_c2m[:3, :3].dot(random_lamp_loc)
            light_diffuse.location = cur_lamp_loc
            light_specular.location = cur_lamp_loc
            light_diffuse.data.energy = np.random.uniform(0.7,0.9)
            light_specular.data.energy = np.random.uniform(0.2,0.4)

            scene.render.image_settings.file_format = 'PNG'  # set output format to .png
            scene.render.filepath =  os.path.join(os.path.join(self.dir_dataset, model_identifier), 'temp_x.png'.format(i))
            bpy.ops.render.render(write_still=True)  # render still

            ####Post process####
            bgra_x = cv2.imread( os.path.join(os.path.join(self.dir_dataset, model_identifier), 'temp_x.png'.format(i)), cv2.IMREAD_UNCHANGED)
            depth_x = bgra_x[:,:,3].copy()
            bgr_x = bgra_x[:, :, 0:3].copy()
            for cc in range(0, 3):
                bgr_x[:, :, cc] = np.where(depth_x, bgr_x[:, :, cc], 255)

            ys, xs = np.nonzero(depth_x > 0)
            obj_bb = view_sampler.calc_2d_bbox(xs, ys, render_dims)
            bbx, bby, bbw, bbh = obj_bb

            crop_offset_sigma = float(20)######Ard 0.15, keep it just, this is an old version of AAE#############
            rand_trans_x = np.random.uniform(-crop_offset_sigma, crop_offset_sigma)
            rand_trans_y = np.random.uniform(-crop_offset_sigma, crop_offset_sigma)

            size = int(np.maximum(bbh,bbw) * pad_factor)
            left = int(bbx + bbw / 2 - size / 2 + rand_trans_x)
            right = int(bbx + bbw / 2 + size / 2 + rand_trans_x)
            top = int(bby + bbh / 2 - size / 2 + rand_trans_y)
            bottom = int(bby + bbh / 2 + size / 2 + rand_trans_y)

            bgr_x = bgr_x[top:bottom, left:right]
            depth_x = depth_x[top:bottom, left:right]
            bgr_x = cv2.resize(bgr_x, self.shape_c3[:2], interpolation=cv2.INTER_NEAREST)
            depth_x = cv2.resize(depth_x, self.shape_c3[:2],  interpolation=cv2.INTER_NEAREST)
            mask_x = depth_x == 0.

            bgra_y = cv2.imread(os.path.join(os.path.join(self.dir_dataset, model_identifier), 'temp_y.png'.format(i)), cv2.IMREAD_UNCHANGED)
            depth_y = bgra_y[:, :, 3].copy()
            bgr_y= bgra_y[:,:,0:3].copy()
            for cc in range(0, 3):
                bgr_y[:, :, cc] = np.where(depth_y, bgr_y[:, :, cc], 255)
            ys, xs = np.nonzero(depth_y > 0)
            obj_bb = view_sampler.calc_2d_bbox(xs, ys, render_dims)

            if debug:
                print(obj_bb,'bpy-272, 87, 92, 87','TLESS-274, 89, 89, 85 rvt-273, 88, 90, 86')

            bgr_y = data_utils.extract_square_patch(bgr_y, obj_bb, pad_factor, resize=self.shape_c3[:2],  interpolation=cv2.INTER_NEAREST)
            depth_y = data_utils.extract_square_patch(depth_y, obj_bb, pad_factor, resize=self.shape_c3[:2],  interpolation=cv2.INTER_NEAREST)
            mask_y = depth_y==0

            self.bgr_x[i] = bgr_x
            self.mask_x[i] = mask_x
            self.bgr_y[i]= bgr_y
            self.mask_y[i] = mask_y
            self.matrix_rot_y[i]=R_m2c

            if i % 100 == 0:
                path_out_dir = os.path.join(self.dir_dataset, model_identifier)
                if not os.path.exists(path_out_dir):
                    os.makedirs(path_out_dir)
                cv2.imwrite(os.path.join(path_out_dir, '{0}_{1}_x_bgr.png'.format(batch_id, i)), self.bgr_x[i])
                cv2.imwrite(os.path.join(path_out_dir, '{0}_{1}_y_bgr.png'.format(batch_id, i)), self.bgr_y[i])

        np.savez(current_file_name, bgr_x=self.bgr_x, mask_x=self.mask_x, bgr_y=self.bgr_y, mask_y=self.mask_y,
                 matrix_rot_y=self.matrix_rot_y)



if __name__=='__main__':
    if True:
        import argparse
        parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
        parser.add_argument('obj', type=str,help='Path to the obj file to be rendered.')
        parser.add_argument('--bid', type=int)
        argv = sys.argv[sys.argv.index("--") + 1:]
        args = parser.parse_args(argv)

        is_shapenet=args.obj.split('.')[-1]=='obj'

        path_splits= args.obj.split('/')
        model_identifier = path_splits[-3]+'_' + path_splits[-2] if is_shapenet else '11'
        render_model=Model(dir_dataset='./tmp/',
                           path_model=args.obj,
                           saved_file_name='prepared_training_data_{:s}_mesh'.format(model_identifier),
                           num_per_batch=5000)
        render_model.render_batch_training_images_blender(render_dims=(640,480),
                                                          cam_K=np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]]).reshape((3,3)),
                                                          batch_id=args.bid,is_shapenet=is_shapenet,debug=False)
        if False:
            render_model.combine_rendered_batches(1)
            render_model.load_training_images()
    else:
        model_id=int(sys.argv[1])
        bid=int(sys.argv[2])
        render_model=Model(dir_dataset='./ws/tmp_datasets/{:02d}'.format(model_id),
                           path_model='./ws/meshes/obj_{:02d}.ply'.format(model_id),
                           saved_file_name='prepared_training_data_{:02d}_hololens'.format(model_id),
                           num_per_batch=10000)
        path_texture={}
        path_texture['01']='./ws/meshes/cherrios_BaseColor.png'
        path_texture['02']='./ws/meshes/CoconutWater_repro.png'
        path_texture['03']='./ws/meshes/Rigatoni_Color_01.png'
        path_texture['04']='./ws/meshes/Spaghetti_Color_02.png'
        print(path_texture['{:02d}'.format(model_id)])
        texture_img_bgr=cv2.imread(path_texture['{:02d}'.format(model_id)])
        texture_img_rgb=texture_img_bgr[:,:,2::-1]
        if False:
            render_model.render_batch_training_images(render_dims=(640,480),
                                                      cam_K=[831.3843876330611,0,640/2,0,831.3843876330611,480/2,0,0,1],
                                                      batch_id=bid,depth_scale=1.,texture_img=texture_img_rgb)
        else:
            render_model.combine_rendered_batches(2)#Combine all generated data into one .npz file
            render_model.load_training_images()#This step is for double check
