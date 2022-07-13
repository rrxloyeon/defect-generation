from rembg import remove
import cv2
import os
from tqdm import tqdm
ALL = False
COMPARE = True
PUT_TEXT = True

input_path = '/home/esoc/datasets/Bulryang_12inch'
output_path = '/home/esoc/koosy/git_rrxloyeon/gen_defect_dataset/defects'
compare_path = '/home/esoc/koosy/git_rrxloyeon/gen_defect_dataset/compared'
class_list = ['Foreign_Material', 'Parasitic'] if not ALL else os.listdir(input_path)
# normal_class = 'Pass'
# bg_path_list = os.listdir(os.path.join(input_path, normal_class))
img_type = '.jpg' # unpacking 이나 머... 다른거 이용해서 파일 확장자 여러개 인 경우 처리하기 (mission)

def hconcat_img(img_list, interpolation = cv2.INTER_CUBIC):
    """
        https://github.com/mafls122/Python_openCV/blob/main/openCV_03.py
    """
    h_min = min(img.shape[0] for img in img_list)
    
    img_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation) for im in img_list]
    
    return cv2.hconcat(img_list_resize)

if COMPARE:
    if not os.path.isdir(os.path.join(compare_path)) : # 이걸 해주지 않으면 cv2가 오류를 발생시키지도 않고 파일도 만들어지지 않음
        os.mkdir(os.path.join(compare_path))
    for c in class_list :
        file_list = os.listdir(os.path.join(input_path, c))
        file_list = [f for f in file_list if img_type in f]
        if not os.path.isdir(os.path.join(compare_path, c)) : # 이걸 해주지 않으면 cv2가 오류를 발생시키지도 않고 파일도 만들어지지 않음
            os.mkdir(os.path.join(compare_path, c))
        
        pbar = tqdm(file_list)
        for img in pbar :
            input = cv2.imread(os.path.join(input_path, c, img))
            output = remove(input)
            if PUT_TEXT:
                loc = (0, input.shape[0]-10)
                input = cv2.putText(input, "original", loc, 0, 1, (0, 0, 255), 2)
                output = cv2.putText(output, "rmbg", loc, 0, 1, (0, 0, 255), 2)
            # input = cv2.cvtColor(input, cv2.COLOR_RGB2RGBA)
            output = cv2.cvtColor(output, cv2.COLOR_RGBA2RGB)
            # output += 255
            output = hconcat_img([input, output])
            cv2.imwrite(os.path.join(compare_path, c, img), output)
            pbar.set_description("Compare! Remove Background " + c)
else:
    if not os.path.isdir(os.path.join(output_path)) : # 이걸 해주지 않으면 cv2가 오류를 발생시키지도 않고 파일도 만들어지지 않음
        os.mkdir(os.path.join(output_path))
    for c in class_list :
        file_list = os.listdir(os.path.join(input_path, c))
        file_list = [f for f in file_list if img_type in f]
        if not os.path.isdir(os.path.join(output_path, c)) : # 이걸 해주지 않으면 cv2가 오류를 발생시키지도 않고 파일도 만들어지지 않음
            os.mkdir(os.path.join(output_path, c))
        
        pbar = tqdm(file_list)
        for img in pbar :
            input = cv2.imread(os.path.join(input_path, c, img))
            output = remove(input)
            output = cv2.cvtColor(output, cv2.COLOR_RGBA2RGB)
            # output += 255
            cv2.imwrite(os.path.join(output_path, c, img), output)
            pbar.set_description("Remove Background " + c)

