import random
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from distribution import guess_background, CLASS_ORANGE, CLASS_GREEN
from albumentations.core.transforms_interface import ImageOnlyTransform

# 나중에 args로 대체
data_dir_args = "/home/esoc/datasets/Bulryang_12inch"
defect_dir_args = "/home/esoc/koosy/git_rrxloyeon/gen_defect/defect"
default_class_args = 'foreign'
border_rate_args = 0.9
save_dir_args = "/home/esoc/koosy/git_rrxloyeon/gen_defect_dataset/madebyme"

class DefectAugmentation(ImageOnlyTransform):
    """
    Impose an image of a defect to the target image
    
    Args:
        defects (int): maximum number of defects to impose
        defects_folder (str): path to the folder with defects images
    """

    def __init__(self, defects=1, dark_defect=False, always_apply=False, p=0.5): # p는 왜 있는거지
        super().__init__(always_apply, p)
        self.defects = defects
        self.dark_defect = dark_defect
        self.defects_folder = defect_dir_args
        self.save_dir = save_dir_args
    
    def make_name(self, classname, num, dir=save_dir_args):
        # make save_fig_name
        if self.dark_defect :
            return "{}/{}/{}.jpg".format(dir+'dark', classname, num)
        else :
            return "{}/{}/{}.jpg".format(dir, classname, num)

    def apply(self, background_dir, class_name):
        """
        Args:
            image (PIL Image): Image to draw defects on.

        Returns:
            PIL Image: Image with drawn defects.
        """
        # get background image ramdomly
        background_images = [im for im in os.listdir(background_dir) if 'jpg' in im]
        src_dir = os.path.join(background_dir, random.choice(background_images))
        image = cv2.cvtColor(cv2.imread(src_dir), cv2.COLOR_BGR2RGB)
        background_type = guess_background(image)
        defects_folder = os.path.join(self.defects_folder, background_type, class_name)
        
        n_defects = random.randint(1, self.defects) # for this example I put 1 instead of 0 to illustrate the augmentation
        
        if not n_defects:
            return image
        
        # height, width, _ = image.shape  # target image width and height
        defects_images = [im for im in os.listdir(defects_folder) if 'png' in im]
        
        defect_dirs = []
        for _ in range(n_defects):
            defect_dirs.append(os.path.join(defects_folder, random.choice(defects_images)))
            defect = cv2.cvtColor(cv2.imread(defect_dirs[-1]), cv2.COLOR_BGR2RGB)
            defect = cv2.flip(defect, random.choice([-1, 0, 1]))
            defect = cv2.rotate(defect, random.choice([0, 1, 2]))
            
            # extract real defect from defect image
            img2gray = cv2.cvtColor(defect, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(img2gray, 254, 255, cv2.THRESH_BINARY_INV)
            mask_inv = cv2.bitwise_not(mask)

            contours,hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            x,y,w,h = cv2.boundingRect(cnt)
            defect = defect[y:y+h,x:x+w]

            h_height, h_width, _ = defect.shape  # defect image width and height
            roi_ho = int(random.gauss((image.shape[0] - defect.shape[0])/2, (image.shape[0] - defect.shape[0])/10))
            roi_wo = int(random.gauss((image.shape[1] - defect.shape[1])/2, (image.shape[0] - defect.shape[0])/10))
            roi = image[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width]

            # Creating a mask and inverse mask 
            img2gray = cv2.cvtColor(defect, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(img2gray, np.mean(defect)*border_rate_args, 255, cv2.THRESH_BINARY_INV)
            mask_inv = cv2.bitwise_not(mask)

            # Now black-out the area of defect in ROI
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

            # Take only region of defect from defect image.
            if self.dark_defect:
                defect_fg = cv2.bitwise_and(img_bg, img_bg, mask=mask)
            else:
                defect_fg = cv2.bitwise_and(defect, defect, mask=mask)

            # Put defect in ROI and modify the target image
            dst = cv2.add(img_bg, defect_fg, dtype=cv2.CV_64F)

            image[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width] = dst
                
        return image, src_dir, defect_dirs, background_type