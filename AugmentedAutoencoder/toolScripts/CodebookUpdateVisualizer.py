import numpy as np
import argparse
import os
import math
import matplotlib.pyplot as plt

argparser = argparse.ArgumentParser()

argparser.add_argument('--path', help='path to the npz file', required=True)
argparser.add_argument('--random', help='will randomly choose samples', action='store_true')
argparser.add_argument('--iteration', help='how many iterations', type=int, default=1)
argparser.add_argument('--sample_size', help='how many samples we used', type=int, default=10000)
argparser.add_argument('--codebook', help='path to the codebook', default='rot_infos_8020.npz')
argparser.add_argument('--knn', help='The k for knn', type=int, default=1)

args = argparser.parse_args()

def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.
    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.
    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    """
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

def convert2quaternions(batch_sample):
    batch_size = batch_sample.shape[0]
    q = np.zeros(shape=(batch_size, 4), dtype=np.float32)
    for i in range(batch_size):
        crot = np.eye(4, dtype=np.float32)
        crot[:3, :3] = batch_sample[i]
        q[i] = quaternion_from_matrix(crot, False)
    return q

if __name__ == '__main__':
    data_path = args.path
    codebook_path = args.codebook
    iteration = args.iteration
    sample_size = args.sample_size
    knn = args.knn

    if not os.path.exists(data_path):
        print ("The input path {} doesn't exist")
        exit()
    
    if not os.path.exists(codebook_path):
        print ("The codebook path {} doesn't exist".format(codebook_path))
        exit()
    
    if sample_size <= 0 or iteration <= 0 or knn <= 0:
        print ("Invalid sample_size or iteration or knn, the program will exit")
        exit()
    
    data = np.load(data_path)
    rot_y = data['matrix_rot_y']
    print ('converting training date to quaternions')
    y_quaternion = convert2quaternions(rot_y)

    # read in codebook
    codebook = np.load(codebook_path)

    rot_matrix = codebook['rots'].astype(np.float32)
    embedding_size = rot_matrix.shape[0]
    print ('converting codebook rotation to quaternions')
    codebook_quaternion = convert2quaternions(rot_matrix)

    for i in range(iteration):
        # random is not implemented yet
        update_times = np.zeros([embedding_size,], dtype=np.int)
        samples = y_quaternion[:sample_size]
        # 10000 x 4 mul 4 x 8020 -> 10000 x 8020
        dot_query_embed = -np.fabs(np.dot(samples, codebook_quaternion.transpose()))
        knn_sample = np.argsort(dot_query_embed, axis=-1)[:, :knn]

        # count the update times
        knn_count = np.unique(knn_sample, return_counts=True)
        update_times[knn_count[0]] += knn_count[1]
        # print (knn_count[0])
        # print (knn_count[1])
        # print (update_times)
        print ("Total {} updates".format(np.sum(knn_count[1])))
        print ("Total {} samples got 0 updates".format(np.sum(update_times==0)))

        # output results.
        update_sort_indice = np.argsort(update_times)[::-1]     # from largest to smallest

        plt.bar(np.arange(update_times.shape[0]), update_times)
        plt.xlabel('Codebook slots')
        plt.ylabel('Update Counts')
        plt.title("Update Counts for each codebook item")
        plt.show()

        # plt
        plt.hist(update_times)
        plt.xlabel('Update Counts')
        plt.ylabel('Numbers')
        plt.title('Update Count Distribution')
        plt.show()



