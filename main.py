from generate_defect import DefectAugmentation
import matplotlib.pyplot as plt
from glob import glob
import pickle5
import cv2
import os
from tqdm import tqdm

PRINT_LOG = True

src_dir_args = "/home/esoc/datasets/Bulryang_12inch/Pass"
save_dir_args = "/home/esoc/koosy/git_rrxloyeon/gen_defect_dataset/madebyme"
save_log_args = True

def make_name(classname, num, dir=save_dir_args):
    # make save_fig_name
    return "{}/{}/{}.jpg".format(dir, classname, num)

def perform_augmentation():
    aug = DefectAugmentation()
    log = {'src_dir' : [], 'defects_dir' : [], 'output_dir':[], 'background_type' : [], 'src_save_dir' : []}
    classes = ['Foreign_Material', 'Parasitic']
    for c in classes :
        files = glob("/home/esoc/koosy/git_rrxloyeon/gen_defect_dataset/original/"+c+"/*")
        # num_files = len([im for im in files if 'jpg' in im])
        num_files = 10
        pbar = tqdm(range(num_files))
        for i in pbar:
            save_fig_name = make_name(c, i)
            image, src_dir, defects_dir, bg_type = aug.apply(src_dir_args, c)
            log['src_dir'].append(src_dir)
            log['defects_dir'].append(defects_dir)
            log['output_dir'].append(save_fig_name)
            log['background_type'].append(bg_type)

            # save figure
            if not os.path.isdir(os.path.split(save_fig_name)[0]):
                os.makedirs(os.path.split(save_fig_name)[0])
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_fig_name, image)
            if PRINT_LOG :
                # print("Success saving image..", c, "{}/{}".format(i, num_files))
                pbar.set_description("Saving Image.. " + c)

            # save figure : Pass(src)
            pass_dir = os.path.join(save_dir_args, 'Pass', bg_type, c+str(i)+'.jpg')
            if not os.path.isdir(os.path.split(pass_dir)[0]):
                os.makedirs(os.path.split(pass_dir)[0])
            try:
                os.system("cp {} {}".format(src_dir, pass_dir))
                log['src_save_dir'].append(pass_dir)
            except:
                log['src_save_dir'].append("cannot cp "+src_dir)
            
            # save log
            if save_log_args :
                with open("log", "wb") as file :
                    pickle5.dump(log, file)

# main
perform_augmentation()