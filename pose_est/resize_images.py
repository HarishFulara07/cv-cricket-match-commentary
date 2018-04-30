import cv2
import glob

_annotated_images_paths = ['dataset/0/*.png', 
        'dataset/1/*.png', 'dataset/2/*.png', 'dataset/3/*.png', 
        'dataset/4/*.png', 'dataset/5/*.png', 'dataset/6/*.png']


for _idx, _img_path in enumerate(_annotated_images_paths):
    _images = [_img for _img in glob.glob(_img_path)]
    for _image in _images:
    	_image_name = _image.split('.')[0].split('/')[2]
    	# print(_image_name)
    	_resized_image = cv2.resize(cv2.imread(_image), (70, 100))
    	cv2.imwrite('dataset/' + str(_idx) + '/' + _image_name + '_resized.png', _resized_image)
