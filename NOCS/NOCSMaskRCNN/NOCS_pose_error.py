import numpy as np

'''
    This is a script of methods to compute pose error for NOCS dataset
'''

def decompose_transform_matrix(M, uniformly_scaling=True):
    assert M.shape[0] == 4
    assert M.shape[1] == 4

    Rm = M[:3, :3]
    T = M[:3, 3]

    # take out S
    if uniformly_scaling:
        S = np.cbrt(np.linalg.det(Rm))
        R = Rm / S
        S = np.array([S,S,S], dtype=M.dtype)
    else:
        S = np.linalg.norm(Rm, axis=0)
        R = Rm / S
    
    return T, S, R

def compute_error_degree_cm(R1, T1, R2, T2, class_name, handle_visibility = 0):
    if class_name in ['bottle', 'can', 'bowl']:  ## symmetric when rotating around y-axis
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    elif class_name == 'mug' and handle_visibility==0:  ## symmetric when rotating around y-axis
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    elif class_name in ['phone', 'eggbox', 'glue']:
        y_180_RT = np.diag([-1.0, 1.0, -1.0])
        R = R1 @ R2.transpose()
        R_rot = R1 @ y_180_RT @ R2.transpose()
        theta = min(np.arccos((np.trace(R) - 1) / 2),
                    np.arccos((np.trace(R_rot) - 1) / 2))
    else:
        R = R1 @ R2.transpose()
        theta = np.arccos((np.trace(R) - 1) / 2)

    theta *= 180 / np.pi
    shift = np.linalg.norm(T1 - T2) * 100
    result = np.array([theta, shift])

    return result

def compute_error_degree_cm_transform(RT1, RT2, class_name, handle_visibility = 0):
    if RT1 is None or RT2 is None:
        return -1
    
    try:
        assert np.array_equal(RT1[3, :], RT2[3, :])
        assert np.array_equal(RT1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(RT1[3, :], RT2[3, :])
        print("The input matrix to compute_AP_degree_cm() is not an affine transform matrix, the last row is not [0,0,0,1]")
        return -1

    T1, _, R1 = decompose_transform_matrix(RT1)
    T2, _, R2 = decompose_transform_matrix(RT2)

    return compute_error_degree_cm(R1, T1, R2, T2, class_name, handle_visibility)
