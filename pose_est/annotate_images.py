import cv2
import glob


def annotate_training_images(_image_dataset_path):
    _images_path = [_img for _img in glob.glob(_image_dataset_path + "/*.png")]

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
            elif _key_pressed == 55:    # 7 Key pressed.
                cv2.imwrite('dataset/7/' + _img_name, _img)
            else:
                continue
            break


annotate_training_images('dataset/test')
