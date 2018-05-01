import cv2
import os
import numpy as np

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
            if count%6 == 0:
                cv2.imwrite(os.path.join(path_output_dir, '%d.png') %(count/6), image)
            count += 1
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()
    return count

def read_image(filename):
    img = cv2.imread(filename,1)
    resized_image = cv2.resize(img, (100, 100)) 
    # cv2.imshow('image',img)
    # cv2.waitKey(5000)
    return resized_image

def generate_color_histogram(image_array):
    image_array/=64
    # print image_array
    colorbin_histogram = np.zeros((64))
    height, width = image_array.shape[:2]
    for i in xrange(height):
        for j in xrange(width):
            index = image_array[i][j][0]*4 + image_array[i][j][1]*2 + image_array[i][j][2]
            colorbin_histogram[index]+=1

    return colorbin_histogram   

def euclidean_distance(hist,next_hist):
    dist=0.0
    for i in xrange(0,64):
        dist+= (hist[i]-next_hist[i])**2
    return dist**0.5

def create_changelist(path_to_directory,lb,ub):
    count = ub-lb+1
    scene_change_array = np.zeros((count))

    for i in xrange(lb,ub):
        filename_cur = os.path.join(path_to_directory,str(i)+'.png')
        filename_next = os.path.join(path_to_directory,str(i+1)+'.png')
        
        img = read_image(filename_cur)
        next_img = read_image(filename_next)
        
        hist = generate_color_histogram(img)
        next_hist = generate_color_histogram(next_img)

        scene_change_array[i] = euclidean_distance(hist,next_hist) 

    return scene_change_array

def segment_shots(scene_change_array):
    for i in range(len(scene_change_array)):
        # print i, scene_change_array[i]
        if(scene_change_array[i]> 1000 and (i==0 or scene_change_array[i] > 5*scene_change_array[i-1])):
            print i+1

count = video_to_frames('video/2016_Final.mp4', 'image/2016_Final/')
scene_change_array = create_changelist('image/2016_Final/',0,count)
segment_shots(scene_change_array)