import cv2
import glob
import numpy 
from sklearn.cluster import MiniBatchKMeans
import random
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import pickle

def splitData():
	vehicles = []
	nonVehicles = []
	for img in glob.glob("Shots/*.png"):
		n= cv2.imread(img,0)
		vehicles.append(n)

	for img in glob.glob("NonShots/*.png"):
		n= cv2.imread(img,0)
		nonVehicles.append(n)

	vehicles = numpy.array(vehicles)
	nonVehicles = numpy.array(nonVehicles)

	numpy.random.shuffle(vehicles)
	numpy.random.shuffle(nonVehicles)

	M = len(vehicles)
	N = len(nonVehicles)

	trainV = vehicles[:0.8*M]
	testV = vehicles[0.8*M:]

	trainnV = nonVehicles[:0.8*N]
	testnV = nonVehicles[0.8*N:]

	return trainV,trainnV,testV,testnV

def getSiftFeatures(trainV,trainnV,testV,testnV):
	sift = cv2.xfeatures2d.SIFT_create()

	trainVSIFT = []
	trainnVSIFT = []
	testVSIFT = []
	testnVSIFT = []

	for i in range(len(trainV)):
		keyPoints, descriptors = sift.detectAndCompute(trainV[i], None)
		if descriptors is not None:
			trainVSIFT.append(descriptors[numpy.random.choice(len(descriptors), size=10, replace=True)])
		else:
			trainVSIFT.append(numpy.zeros((10,128)))

	for i in range(len(testV)):
		keyPoints, descriptors = sift.detectAndCompute(testV[i], None)
		if descriptors is not None:
			testVSIFT.append(descriptors[numpy.random.choice(len(descriptors), size=10, replace=True)])
		else:
			testVSIFT.append(numpy.zeros((10,128)))

	for i in range(len(trainnV)):
		keyPoints, descriptors = sift.detectAndCompute(trainnV[i], None)
		if descriptors is not None:
			trainnVSIFT.append(descriptors[numpy.random.choice(len(descriptors), size=10, replace=True)])
		else:
			trainnVSIFT.append(numpy.zeros((10,128)))

	for i in range(len(testnV)):
		keyPoints, descriptors = sift.detectAndCompute(testnV[i], None)
		if descriptors is not None:
			testnVSIFT.append(descriptors[numpy.random.choice(len(descriptors), size=10, replace=True)])
		else:
			testnVSIFT.append(numpy.zeros((10,128)))

	return trainVSIFT,trainnVSIFT,testVSIFT,testnVSIFT

def getKMeans(vSIFT, nvSIFT):
	featureDescriptors = []
	for i in range(0,len(vSIFT)):
		for j in range(10):
			featureDescriptors.append(vSIFT[i][j])
	for i in range(len(nvSIFT)):
		for j in range(10):
			featureDescriptors.append(nvSIFT[i][j])

	kmeans = MiniBatchKMeans(100).fit(featureDescriptors)
	labels =  kmeans.labels_
	cluster_centers = kmeans.cluster_centers_

	vImageVecs = []
	vnImageVecs = []

	for i in range(len(vSIFT)):
		t = labels[10*i:10*i+10]
		z = numpy.zeros(100)
		for j in range(10):
			z[t[j]]+=1
		vImageVecs.append(z)

	for i in range(len(vSIFT),len(vSIFT)+len(nvSIFT)):
		t = labels[10*i:10*i+10]
		z = numpy.zeros(100)
		for j in range(10):
			z[t[j]]+=1
		vnImageVecs.append(z)

	return vImageVecs, vnImageVecs, kmeans

def svmClassifier(vImageVecs,vnImageVecs):
	combinedImageDescriptors = vImageVecs + vnImageVecs
	M = len(vImageVecs)
	N = len(vnImageVecs)

	Y = numpy.zeros((M+N))
	Y[M:] = 1

	# params = {'kernel':('linear'), 'C':[50]}
	clf = SVC(0.0001)
	# clf = GridSearchCV(clf, params)
	clf.fit(combinedImageDescriptors, Y) 

	return clf

def testImages(clf,kmeans,testVSIFT,testnVSIFT):
	score = 0.0
	testArr = testVSIFT+ testnVSIFT

	testVectors = []
	for i in range(len(testArr)):
		descriptor = kmeans.predict(testArr[i])
		z = numpy.zeros(100)
		for j in range(10):
			z[descriptor[j]]+=1
		testVectors.append(z)

	# print testArr
	print len(testVectors),len(testVectors[0])
	class_p = clf.predict(testVectors)

	l = len(testVSIFT)

	for i in range(len(testVSIFT)):
		if(class_p[i]==0):
			score+=1

	for i in range(len(testnVSIFT)):
		if(class_p[l+i]==1):
			score+=1

	TOT = len(testVSIFT) + len(testnVSIFT)
	score = score/(TOT*1.0)
	return score


trainV,trainnV,testV,testnV = splitData()
print 'data split'

trainVSIFT,trainnVSIFT,testVSIFT,testnVSIFT = getSiftFeatures(trainV,trainnV,testV,testnV)
print 'SIFT features computed'

vImageVecs, vnImageVecs, kmeans = getKMeans(trainVSIFT,trainnVSIFT)
print 'Got vector corresponding each image'

clf = svmClassifier(vImageVecs,vnImageVecs)
print 'Trained SVM Model with 5 fold cross validation'

# filename = 'model/clutter_model.sav'
# pickle.dump(clf, open(filename, 'wb'))

# clf = pickle.load(open(filename, 'rb'))

score = testImages(clf,kmeans,testVSIFT,testnVSIFT)
print 'Tested model on test dataset'
print score