import cv2
import glob


def annotate_training_images(_image_dataset_path):
    _images_path = [_img for _img in glob.glob(_image_dataset_path + "/*.png")]

    _annotated_images_paths = ['dataset/-1/*.png', 'dataset/0/*.png', 
        'dataset/1/*.png', 'dataset/2/*.png', 'dataset/3/*.png', 
        'dataset/4/*.png', 'dataset/5/*.png', 'dataset/6/*.png']

    # Don't annotate the already annotated images.
    for _img_path in _annotated_images_paths:
        _annotated_images_name = [_img.split('/')[-1] for _img in glob.glob(_img_path)]
        for _image_name in _annotated_images_name:
            _image_name = 'dataset/test/' + _image_name
            _images_path.remove(_image_name)

    for _image_path in _images_path:
        _img = cv2.imread(_image_path)
        while(1):
            cv2.imshow('Training Image', _img)
            _key_pressed = cv2.waitKey(33)
            _img_name = _image_path.split('/')[-1]
            if _key_pressed == 48:    # 0 Key pressed.
                cv2.imwrite('dataset/0/' + _img_name, _img)
            elif _key_pressed == 49:    # 1 Key pressed.
                cv2.imwrite('dataset/1/' + _img_name, _img)
            elif _key_pressed == 50:    # 2 Key pressed.
                cv2.imwrite('dataset/2/' + _img_name, _img)
            elif _key_pressed == 51:    # 3 Key pressed.
                cv2.imwrite('dataset/3/' + _img_name, _img)
            elif _key_pressed == 52:    # 4 Key pressed.
                cv2.imwrite('dataset/4/' + _img_name, _img)
            elif _key_pressed == 53:    # 5 Key pressed.
                cv2.imwrite('dataset/5/' + _img_name, _img)
            elif _key_pressed == 54:    # 6 Key pressed.
                cv2.imwrite('dataset/6/' + _img_name, _img)
            elif _key_pressed == 55:    # 7 Key pressed. These are wrong/bad images.
                cv2.imwrite('dataset/-1/' + _img_name, _img)
            else:
                continue
            break


annotate_training_images('dataset/test')
