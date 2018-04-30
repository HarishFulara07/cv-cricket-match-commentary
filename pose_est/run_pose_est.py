import matlab.engine
import glob
import numpy as np
import h5py


def full_pose_estimation(_eng, _image_dataset_root_dir, _image_name):
    try:
        _pose_mat, _height, _width = _eng.run_full(_image_dataset_root_dir, _image_name, 0, nargout=3)
        _pose_mat_x = np.array(_pose_mat[0][::2]) / _width
        _pose_mat_y = np.array(_pose_mat[0][1::2]) / _height
        return np.hstack((np.vstack(_pose_mat_x), np.vstack(_pose_mat_y)))
    except matlab.engine.MatlabExecutionError:
        return None

def create_pose_features_dataset(_features_dataset_name, _labels_dataset_name):
    print('Starting matlab engine...')
    _eng = matlab.engine.start_matlab()
    print('Matlab engine started...')

    _shot_0_images_name = [_img.split('/')[-1] for _img in glob.glob('dataset/0/' + "*.png")]
    _shot_1_images_name = [_img.split('/')[-1] for _img in glob.glob('dataset/1/' + "*.png")]
    _shot_2_images_name = [_img.split('/')[-1] for _img in glob.glob('dataset/2/' + "*.png")]
    _shot_3_images_name = [_img.split('/')[-1] for _img in glob.glob('dataset/3/' + "*.png")]
    _shot_4_images_name = [_img.split('/')[-1] for _img in glob.glob('dataset/4/' + "*.png")]
    _shot_5_images_name = [_img.split('/')[-1] for _img in glob.glob('dataset/5/' + "*.png")]
    _shot_6_images_name = [_img.split('/')[-1] for _img in glob.glob('dataset/6/' + "*.png")]

    _total_images = (len(_shot_0_images_name) + len(_shot_1_images_name) \
            + len(_shot_2_images_name) + len(_shot_3_images_name) \
            + len(_shot_4_images_name) + len(_shot_5_images_name) \
            + len(_shot_6_images_name)) / 2

    print('Total images: {}'.format(_total_images))

    _feature_pose_mat = np.zeros((_total_images, 52, 2))
    _shot_labels = np.zeros(_total_images, dtype=np.int8)
    _idx = 0

    for _image_name in _shot_0_images_name:
        if 'resized' in _image_name:
            tmp = full_pose_estimation(_eng, '0', _image_name)
            if tmp is not None:
                _feature_pose_mat[_idx] = tmp
                _idx += 1

    for _image_name in _shot_1_images_name:
        if 'resized' in _image_name:
            tmp = full_pose_estimation(_eng, '1', _image_name)
            if tmp is not None:
                _feature_pose_mat[_idx] = tmp
                _shot_labels[_idx] = 1
                _idx += 1

    for _image_name in _shot_2_images_name:
        if 'resized' in _image_name:
            tmp = full_pose_estimation(_eng, '2', _image_name)
            if tmp is not None:
                _feature_pose_mat[_idx] = tmp
                _shot_labels[_idx] = 2
                _idx += 1

    for _image_name in _shot_3_images_name:
        if 'resized' in _image_name:
            tmp = full_pose_estimation(_eng, '3', _image_name)
            if tmp is not None:
                _feature_pose_mat[_idx] = tmp
                _shot_labels[_idx] = 3
                _idx += 1

    for _image_name in _shot_4_images_name:
        if 'resized' in _image_name:
            tmp = full_pose_estimation(_eng, '4', _image_name)
            if tmp is not None:
                _feature_pose_mat[_idx] = tmp
                _shot_labels[_idx] = 4
                _idx += 1

    for _image_name in _shot_5_images_name:
        if 'resized' in _image_name:
            tmp = full_pose_estimation(_eng, '5', _image_name)
            if tmp is not None:
                _feature_pose_mat[_idx] = tmp
                _shot_labels[_idx] = 5
                _idx += 1

    for _image_name in _shot_6_images_name:
        if 'resized' in _image_name:
            tmp = full_pose_estimation(_eng, '6', _image_name)
            if tmp is not None:
                _feature_pose_mat[_idx] = tmp
                _shot_labels[_idx] = 6
                _idx += 1

    # The final data that will be used for training.
    _h5f = h5py.File('dataset/data.h5', 'w')
    _h5f.create_dataset(_features_dataset_name, data=_feature_pose_mat)
    _h5f.create_dataset(_labels_dataset_name, data=_shot_labels)
    _h5f.close()


create_pose_features_dataset('train_features', 'train_labels')

# h5f = h5py.File('dataset/data.h5','r')
# a = h5f['train_features'][:]
# b = h5f['train_labels'][:]
# h5f.close()
# print(a)
# print(b)
