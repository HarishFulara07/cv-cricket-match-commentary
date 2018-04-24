import matlab.engine
import glob
import numpy as np
import h5py


def create_pose_features_dataset(_image_dataset_path, _feature_dataset_name):
    _images_name = [_img.split('/')[-1] for _img in glob.glob(_image_dataset_path + "/*.png")]

    print('Starting matlab engine...')
    _eng = matlab.engine.start_matlab()
    print('Matlab engine started...')

    _image_dataset_root_dir = _image_dataset_path.split('/')[-1]
    _feature_pose_mat = np.zeros((len(_images_name), 52, 2))
    
    for _idx, _image_name in enumerate(_images_name):
        _pose_mat, _height, _width = _eng.run_full(_image_dataset_root_dir, _image_name, 0, nargout=3)
        _pose_mat_x = np.array(_pose_mat[0][::2]) / _width
        _pose_mat_y = np.array(_pose_mat[0][1::2]) / _height
        _feature_pose_mat[_idx] = np.hstack((np.vstack(_pose_mat_x), np.vstack(_pose_mat_y)))

    h5f = h5py.File('dataset/data.h5', 'w')
    h5f.create_dataset(_feature_dataset_name, data=_feature_pose_mat)
    h5f.close()


# create_pose_features_dataset('dataset/test', 'train')
