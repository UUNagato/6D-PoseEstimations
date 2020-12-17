'''
    Script used to split NOCS transformation matrix RTs to unit orthogonal matrix, translation vector and scale vector.
'''

import pickle
import numpy as np
import os
import glob
import argparse

def decompose_RT(RTs):
    assert (RTs.ndim == 3)
    RSMatrix = RTs[:,:3,:3]
    SMatrix = np.linalg.norm(RSMatrix, axis=1)
    RMatrix = RSMatrix / SMatrix[:,:,np.newaxis]
    TMatrix = RTs[:,:3,3]
    return RMatrix, SMatrix, TMatrix

def obj2cam_fromworld(RTs):
    m2c_matrix = np.matmul(np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]), RTs)
    return m2c_matrix

def process_pkl_file(file_path, flip_z = False):
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        print ('pkl file:{} does not exist'.format(file_path))
        return None
    
    with open(file_path, 'rb') as f:
        try:
            data = pickle.load(f)
            if 'pred_RTs' not in data.keys():
                print ("Wrong format of pkl file {}, there is no key called pred_RTs".format(file_path))
                return None
            if 'pred_o2c_RTs' in data.keys():
                print ("File {} is skipped since there is already obj2cam RTs".format(file_path))
            RTs = data['pred_RTs']
            R, S, T = decompose_RT(RTs)

            data['pred_Rs'] = R
            data['pred_Ts'] = T
            data['pred_Ss'] = S

            # obj2cam
            RT_o2c = obj2cam_fromworld(RTs)
            o2cR, o2cS, o2cT = decompose_RT(RT_o2c)

            data['pred_o2c_RTs'] = RT_o2c
            data['pred_o2c_Rs'] = o2cR
            data['pred_o2c_Ts'] = o2cT
            data['pred_o2c_Ss'] = o2cS

            gtRTs = data['gt_RTs']
            gtR, gtS, gtT = decompose_RT(gtRTs)
            data['gt_Rs'] = gtR
            data['gt_Ts'] = gtT
            data['gt_Ss'] = gtS

            gtRT_o2c = obj2cam_fromworld(gtRTs)
            gto2cR, gto2cS, gto2cT = decompose_RT(gtRT_o2c)

            data['gt_o2c_RTs'] = gtRT_o2c
            data['gt_o2c_Rs'] = gto2cR
            data['gt_o2c_Ts'] = gto2cT
            data['gt_o2c_Ss'] = gto2cS

            # try to save
            try:
                split_file_path = os.path.split(file_path)
                dump_file_name = split_file_path[1]
                dump_file_name = dump_file_name[:dump_file_name.rfind('.')]
                dump_file_name = dump_file_name + '_o2c.pkl'
                complete_path = os.path.join(split_file_path[0], dump_file_name)
                print ("Save processed data to {}".format(complete_path))
                with open(complete_path, 'wb') as outf:
                    pickle.dump(data, outf)
            except Exception as e:
                print ("Failed to output processed data for file:{}".format(file_path))

        except Exception as e:
            print ('Failed to process file:{}'.format(file_path))
            print ("Exception:{}".format(e))
        
def process_pkls_infolder(folder, flip_z = False):
    if not os.path.isdir(folder):
        print ('the input {} is not a folder'.format(folder))
        return
    
    process_glob = os.path.join(folder, "results_*_*_*.pkl")
    pkl_files = glob.glob(process_glob)

    for pkl in pkl_files:
        print ('processing file {}'.format(pkl))
        process_pkl_file(pkl)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="the path to prediction files", required=True)
    parser.add_argument("--flip_z", help="If the z axis should be flipped (It doesn't work for now)", default="False")

    arguments = parser.parse_args()

    folder = arguments.data
    process_pkls_infolder(folder)

if __name__ == '__main__':
    main()
