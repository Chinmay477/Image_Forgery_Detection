import torchvision.transforms.functional as tf
import pandas as pd
import os
import numpy as np
from skimage.util import view_as_windows
from skimage import io
import PIL


def get_ref_df():

    refs1 = pd.read_csv('../data/NC2016/reference/manipulation/NC2016-manipulation-ref.csv',
                        delimiter='|')
    refs2 = pd.read_csv('../data/NC2016/reference/removal/NC2016-removal-ref.csv', delimiter='|')
    refs3 = pd.read_csv('../data/NC2016/reference/splice/NC2016-splice-ref.csv', delimiter='|')
    all_refs = pd.concat([refs1, refs2, refs3], axis=0)
    return all_refs


def delete_prev_images(dir_name):

    for the_file in os.listdir(dir_name):
        file_path = os.path.join(dir_name, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def check_and_reshape(image, input_mask):

    try:
        mask_x, mask_y = input_mask.shape
        mask = np.empty((mask_x, mask_y, 1))
        mask[:, :, 1] = input_mask
    except ValueError:
        mask = input_mask
    if image.shape == mask.shape:
        return image, mask
    elif image.shape[0] == mask.shape[1] and image.shape[1] == mask.shape[0]:
        mask = np.reshape(mask, (image.shape[0], image.shape[1], mask.shape[2]))
        return image, mask
    else:
        return image, mask


def extract_all_patches(image, window_shape, stride, num_of_patches, rotations, output_path, im_name, rep_num, mode):

    non_tampered_windows = view_as_windows(image, window_shape, step=stride)
    non_tampered_patches = []
    for m in range(non_tampered_windows.shape[0]):
        for n in range(non_tampered_windows.shape[1]):
            non_tampered_patches += [non_tampered_windows[m][n][0]]
    save_patches(non_tampered_patches, num_of_patches, mode, rotations, output_path, im_name, rep_num,
                 patch_type='authentic')


def save_patches(patches, num_of_patches, mode, rotations, output_path, im_name, rep_num, patch_type):

    inds = np.random.choice(len(patches), num_of_patches, replace=False)
    if mode == 'rot':
        for i, ind in enumerate(inds):
            image = patches[ind][0] if patch_type == 'tampered' else patches[ind]
            for angle in rotations:
                im_rt = tf.rotate(PIL.Image.fromarray(np.uint8(image)), angle=angle,
                                  resample=PIL.Image.BILINEAR)
                im_rt.save(output_path + '/{0}/{1}_{2}_{3}_{4}.png'.format(patch_type, im_name, i, angle, rep_num))
    else:
        for i, ind in enumerate(inds):
            image = patches[ind][0] if patch_type == 'tampered' else patches[ind]
            io.imsave(output_path + '/{0}/{1}_{2}_{3}.png'.format(patch_type, im_name, i, rep_num), image)


def create_dirs(output_path):

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(output_path + '/authentic')
        os.makedirs(output_path + '/tampered')
    else:
        if os.path.exists(output_path + '/authentic'):
            delete_prev_images(output_path + '/authentic')
        else:
            os.makedirs(output_path + '/authentic')
        if os.path.exists(output_path + '/tampered'):
            delete_prev_images(output_path + '/tampered')
        else:
            os.makedirs(output_path + '/tampered')


class NotSupportedDataset(Exception):
    pass


def find_tampered_patches(image, im_name, mask, window_shape, stride, dataset, patches_per_image):

    patches = view_as_windows(image, window_shape, step=stride)

    if dataset == 'casia2':
        mask_patches = view_as_windows(mask, window_shape, step=stride)
    elif dataset == 'nc16':
        mask_patches = view_as_windows(mask, (128, 128), step=stride)
    else:
        raise NotSupportedDataset('The datasets supported are casia2 and nc16')

    tampered_patches = []

    for m in range(patches.shape[0]):
        for n in range(patches.shape[1]):
            im = patches[m][n][0]
            ma = mask_patches[m][n][0]
            num_zeros = (ma == 0).sum()
            num_ones = (ma == 255).sum()
            total = num_ones + num_zeros
            if dataset == 'casia2':
                if num_zeros <= 0.99 * total:
                    tampered_patches += [(im, ma)]
            elif dataset == 'nc16':
                if 0.80 * total >= num_ones >= 0.20 * total:
                    tampered_patches += [(im, ma)]

    num_of_patches = patches_per_image
    if len(tampered_patches) < num_of_patches:
        print("Number of tampered patches for image {} is only {}".format(im_name, len(tampered_patches)))
        num_of_patches = len(tampered_patches)

    return tampered_patches, num_of_patches
