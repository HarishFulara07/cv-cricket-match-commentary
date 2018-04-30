from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

import h5py

if __name__ == '__main__':
	h5f = h5py.File('dataset/data.h5','r')
	X = h5f['train_features'][:]
	y = h5f['train_labels'][:]
	n_samples, nx, ny = X.shape
	X = X.reshape((n_samples, nx*ny))
	h5f.close()

	X_train, X_test, y_train, y_test = train_test_split(X, y, 
		train_size=0.7, test_size=0.3, random_state=7)

	_n_estimators_params = [10, 20, 30]
	_max_depth_params = [None, 5, 10]
	_min_samples_split_params = [2, 5, 10]
	_skf = StratifiedKFold(n_splits=5)
	_rf_params = [{'n_estimators': _n_estimators_params, 'max_depth': _max_depth_params,
		'min_samples_split': _min_samples_split_params}]
	_clf = GridSearchCV(RandomForestClassifier(), _rf_params, cv=_skf, scoring='f1_weighted', n_jobs=2)
	# _clf = SVC(kernel='linear')
	_clf.fit(X_train, y_train)
	_scores = _clf.cv_results_['mean_test_score']
	print('F1 Scores: {}'.format(_scores))

	_best_clf = _clf.best_estimator_
	y_pred = _best_clf.predict(X_test)
	# y_pred = _clf.predict(X_test)
	print(classification_report(y_test, y_pred))
