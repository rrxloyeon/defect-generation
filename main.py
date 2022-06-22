from generate_defect import DefectAugmentation
import matplotlib.pyplot as plt

src_dir_args = "/home/esoc/datasets/Bulryang_12inch/Pass"
save_fig_name = "test.jpg"

aug = DefectAugmentation()
image, src_dir, defects_dir = aug.apply()

print("koo test : src_dir :", src_dir)
print("koo test : defects_dir :")
for d in defects_dir :
    print(d)
plt.imshow(image)
plt.savefig(save_fig_name)
print("Success saving image..", save_fig_name)