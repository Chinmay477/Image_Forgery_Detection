import os
from glob import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity


def find_mask(sp_pic, au_pic_dict):

    background_index = [13, 21]
    save_name = sp_pic.split(os.sep)[-1][:-4]
    sp_name = sp_pic.split(os.sep)[-1][background_index[0]:background_index[1]]
    if sp_name in au_pic_dict.keys():
        au_pic = au_pic_dict[sp_name]
        au_image = plt.imread(au_pic)
        sp_image = plt.imread(sp_pic)
        if sp_image.shape == au_image.shape:

            gray_au_image = cv2.cvtColor(au_image, cv2.COLOR_BGR2GRAY)
            gray_sp_image = cv2.cvtColor(sp_image, cv2.COLOR_BGR2GRAY)

            (_, diff) = structural_similarity(gray_au_image, gray_sp_image, full=True)
            diff = cv2.medianBlur(diff, 1)

            mask = np.ones_like(diff)
            mask[diff < 0.98] = 1
            mask[diff >= 0.98] = 0
            mask = (mask * 255).astype("uint8")
            cv2.imwrite('masks/' + save_name + '_gt.png', mask)


def extract_masks():
    """
    Extracts and saves all the masks.
    """

    save_dir = 'masks'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    au_pic_list = glob('..' + os.sep + '..' + os.sep + 'data' + os.sep + 'CASIA2' + os.sep + 'Au' + os.sep + '*')
    sp_pic_list = glob('..' + os.sep + '..' + os.sep + 'data' + os.sep + 'CASIA2' + os.sep + 'Tp' + os.sep + '*')

    au_index = [3, 6, 7, 12]

    au_pic_dict = {
        au_pic.split(os.sep)[-1][au_index[0]:au_index[1]] + au_pic.split(os.sep)[-1][au_index[2]:au_index[3]]:
            au_pic for au_pic
        in au_pic_list}
    for _, Sp_pic in enumerate(sp_pic_list):
        find_mask(Sp_pic, au_pic_dict)


if __name__ == '__main__':
    extract_masks()
