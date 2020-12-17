import numpy as np
import pickle
import argparse
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--dir", required=True, default=None)

args = parser.parse_args()

def checkRotMatrix(m, threshold = 1e-5):
    R = m[:3, :3]
    s = np.linalg.norm(R, axis=0)

    diff1 = abs(s[0] - s[1])
    diff2 = abs(s[0] - s[2])
    diff3 = abs(s[1] - s[2])

    if diff1 < threshold and diff2 < threshold and diff3 < threshold:
        return True, None
    else:
        return False, (diff1, diff2, diff3)


if __name__ == '__main__':
    dir = args.dir

    if not os.path.isdir(dir):
        print ('The input path {} is not a folder path'.format(dir))
        exit()
    
    file_exp = 'results_*.pkl'
    file_list = glob.glob(os.path.join(dir, file_exp))

    print ('total {} files are detected.'.format(len(file_list)))

    for pkl_file in file_list:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                if 'gt_RTs' not in data:
                    print ("File {} doesn't have a key gt_RTs, so it's skipped".format(pkl_file))
                    continue

                RTs = data['gt_RTs']
                for RT in RTs:
                    ret, s = checkRotMatrix(RT)
                    if not ret:
                        print ("File {} failed to pass the uniformly scaling test on gt_RTs. The data is:\n{}\n, The diff is:\n{}".format(pkl_file, RT, s))
                
                if 'pred_RTs' not in data:
                    print ("File {} doesn't have a key pred_RTs, so it's skipped".format(pkl_file))
                    continue

                RTs = data['pred_RTs']
                for RT in RTs:
                    ret, s = checkRotMatrix(RT)
                    if not ret:
                        print ("File {} failed to pass the uniformly scaling test on pred_RTs. The data is:\n{}\n, The diff is:\n{}".format(pkl_file, RT, s))
        except Exception as e:
            print ('An exception happened with file:{}, the exception is:{}'.format(pkl_file, e))

