import cv2
import pickle5
import os

def hconcat_img(img_list, interpolation = cv2.INTER_CUBIC):
    """
        https://github.com/mafls122/Python_openCV/blob/main/openCV_03.py
    """
    h_min = min(img.shape[0] for img in img_list)
    
    img_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation) for im in img_list]

    return cv2.hconcat(img_list_resize)  # 결합해서 리턴

ori_datadir = '/home/esoc/datasets/Bulryang_12inch'
log_file_list = [f for f in os.listdir('./') if 'log_' in f]

for log_file in log_file_list:
    with open(log_file, 'rb') as f:
        log = pickle5.load(f)

    for i, defects_dir in enumerate(log['defects_dir']):
        class_name = defects_dir[0].split(os.path.sep)[-2]
        
        # read pure image
        pure_image_name = os.path.basename(defects_dir[0])[:-4]+'.jpg'
        pure_image_path = os.path.join(ori_datadir, class_name, pure_image_name)
        pure_image = cv2.imread(pure_image_path)

        # read generated image
        gen_image = cv2.imread(log['output_dir'][i])

        # concatenate
        print(i, class_name, pure_image_name)
        print(log['output_dir'][i])
        print("pure", pure_image.shape)
        print("gen", gen_image.shape)
        if pure_image.shape != gen_image.shape:
            image = hconcat_img([pure_image, gen_image])
        else:
            image = cv2.hconcat([pure_image, gen_image])

        # wirte compared image
        write_dir = os.path.join('./compared', log_file, class_name)
        if not os.path.isdir(write_dir):
            os.makedirs(write_dir)
        cv2.imwrite(write_dir+'/{}.jpg'.format(i), image)
