import os
import glob
import shutil

_annotated_images_paths = ['dataset/0/*.png', 
        'dataset/1/*.png', 'dataset/2/*.png', 'dataset/3/*.png', 
        'dataset/4/*.png', 'dataset/5/*.png', 'dataset/6/*.png']

for _idx, _img_path in enumerate(_annotated_images_paths):
    _annotated_images_name = [_img.split('/')[-1] for _img in glob.glob(_img_path)]
    # Some images have a dot (.) in their name. Correct those to dash (-)
    for _image_name in _annotated_images_name:
    	_orig_image_name = _image_name
    	_split_image_name = _image_name.split('.')
    	if len(_split_image_name) > 2:
        	_split_image_name.remove('png')
        	_image_name = '-'.join(_split_image_name)
        	os.rename('dataset/' + str(_idx) + '/' + _orig_image_name,
        		'dataset/' + str(_idx) + '/' + _image_name + '.png')
    # Some images have a spaces in their name. Correct those to underscore (_)
    for _image_name in _annotated_images_name:
        _orig_image_name = _image_name
        _split_image_name = _image_name.split(' ')
        if len(_split_image_name) > 2:
            _image_name = '_'.join(_split_image_name)
            # print('dataset/' + str(_idx) + '/' + _orig_image_name)
            # print('dataset/' + str(_idx) + '/' + _image_name)
            os.rename('dataset/' + str(_idx) + '/' + _orig_image_name,
                'dataset/' + str(_idx) + '/' + _image_name)
