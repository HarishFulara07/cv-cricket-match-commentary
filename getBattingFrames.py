import cv2
import os
import numpy as np
from matplotlib import pyplot as plt


global_threshold = 0.54
def getCroppedImages(imageFrame,top_left,bottom_right,videoFolderName,index):
    imageCrop = imageFrame.copy()
    W = bottom_right[0] - top_left[0]
    H = bottom_right[1] - top_left[1]
    X0 = top_left[0]
    Y0 = top_left[1]
    imageCrop = imageCrop[Y0:Y0+H , X0:X0+W ]
    cv2.imshow('Image.png',imageCrop)
    cv2.imwrite('Obtained/'+str(index)+'.png',imageCrop)
    print 'index: ', index

def video_to_frames(video, path_output_dir):
    # extract frames from a video and save to directory as 'x.png' where 
    # x is the frame index
    vidcap = cv2.VideoCapture(video)
    count = 0
    while vidcap.isOpened():
        success, image = vidcap.read()
        # print count
        if success:
            # print '-'
            if count%3 == 0:
                cv2.imwrite(os.path.join(path_output_dir, '%d.png') %(count), image)
            count += 1
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()
    return count

def findActionBox(frameName,templateFolder,numTemplates,videoFolderName,ind):
    print frameName
    imageFrame = cv2.imread(frameName,1)
    grayFrame = cv2.cvtColor(imageFrame,cv2.COLOR_BGR2GRAY)
    imShape = grayFrame.shape

    templatesNames = [os.path.join(templateFolder,str(i)+'.png') for i in range(1,numTemplates+1)]
    

    templateImages = [cv2.imread(templatesNames[i],0) for i in range(0,numTemplates)]
    templateShape = [templateImages[i].shape[::-1] for i in range(0,numTemplates)]

    mask = np.array([cv2.matchTemplate(grayFrame,templateImages[i],cv2.TM_CCOEFF_NORMED) for i in range(numTemplates)])
    # locations = [np.where(mask[i] > global_threshold) for i in range(numTemplates)]
    correlation = [np.amax(mask[i]) for i in range(numTemplates)]
    locations = [np.argmax(mask[i]) for i in range(numTemplates)]
    index = np.argmax(correlation)
    print index,correlation[index]
    if(correlation[index] > global_threshold):
        imageIndex = locations[index]
        matchingTemplateShape = templateShape[index]
        # print imageIndex,matchingTemplateShape
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(mask[index])
        top_left = max_loc
        bottom_right = (top_left[0] + matchingTemplateShape[0], top_left[1] + matchingTemplateShape[1])
        cv2.rectangle(imageFrame,top_left, bottom_right, 255, 2)
        getCroppedImages(imageFrame,top_left,bottom_right,videoFolderName,ind)
        # plt.imshow(imageFrame)
        # plt.show()
        # return 1
    else:

        # plt.imshow(imageFrame)
        # plt.show()
        print 'No Suitable Matching Found'
        # return 0

    cv2.imshow('Frame.jpg',imageFrame)
    cv2.waitKey(1)

def actionFrameCover(count,videoFolderName,numberOfCrops):
    for i in range(count):
        print i
        ret = findActionBox('image/'+videoFolderName+'/'+str(3*i)+'.png','Crops',numberOfCrops,videoFolderName,i)

videoFolderName = 'Bangladesh720'
count = 10773/3
numberOfCrops = 31
# count = video_to_frames('video/'+videoFolderName+'.mp4', 'image/'+videoFolderName+'/')
actionFrameCover(count,videoFolderName,numberOfCrops)