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


if __name__ == '__main__':
	_images = [_img for _img in glob.glob('dataset/final/*.png')]
	_eng = matlab.engine.start_matlab()
	for _image in _images:
		_image_name = _image.split('/')[-1]
		_pose_mat = full_pose_estimation(_eng, 'final', _image_name)
		nx, ny = _pose_mat.shape
		_pose_mat = _pose_mat.reshape((1, nx*ny))
		if _pose_mat is not None:
			print(_pose_mat.shape)
			_clf = joblib.load('dataset/svm.pkl')
			_shot_label = _clf.predict(_pose_mat)
			print(_image_name + " : " + str(_shot_label[0]))
		else:
			print('Unable to detect pose')
