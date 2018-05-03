from sklearn.externals import joblib
import glob
import matlab.engine
import numpy as np


def full_pose_estimation(_eng, _image_dataset_root_dir, _image_name):
    try:
        _pose_mat, _height, _width = _eng.run_full(_image_dataset_root_dir, _image_name, 0, nargout=3)
        _pose_mat_x = np.array(_pose_mat[0][::2]) / _width
        _pose_mat_y = np.array(_pose_mat[0][1::2]) / _height
        return np.hstack((np.vstack(_pose_mat_x), np.vstack(_pose_mat_y)))
    except matlab.engine.MatlabExecutionError:
        return None


def shot_name(_shot_label):
	if _shot_label == 0:
		return "No shot"
	elif _shot_label == 1:
		return "Cover Drive"
	elif _shot_label == 2:
		return "Straight Drive"
	elif _shot_label == 3:
		return "Pull/Hook Shot"
	elif _shot_label == 4:
		return "Cut Shot"
	elif _shot_label == 5:
		return "Off Drive Shot"
	elif _shot_label == 6:
		return "On Drive Shot"


def run_pipeline(_action_box_img_path):
	_eng = matlab.engine.start_matlab()
	_clf = joblib.load('dataset/svm.pkl')
	_img_name = _action_box_img.split('/')[-1]
	_pose_mat = full_pose_estimation(_eng, 'final', _img_name)
	if _pose_mat is not None:
		_nx, _ny = _pose_mat.shape
		_pose_mat = _pose_mat.reshape((1, _nx*_ny))
		_shot_label = _clf.predict(_pose_mat)
		return shot_name(_shot_label)
	else:
		return -1
