from skimage import io
import warnings

from .extraction_utils import get_ref_df, check_and_reshape, extract_all_patches, create_dirs, save_patches, \
    find_tampered_patches

# from src.patch_extraction.mask_extraction import extract_masks
warnings.filterwarnings('ignore')


class PatchExtractorNC:


    def __init__(self, input_path, output_path, patches_per_image=4, rotations=8, stride=8, mode='no_rot'):

        self.patches_per_image = patches_per_image
        self.stride = stride
        rots = [0, 90, 180, 270]
        self.rotations = rots[:rotations]
        self.mode = mode
        self.input_path = input_path
        self.output_path = output_path

    def extract_authentic_patches(self, d, num_of_patches, rep_num):

        window_shape = (128, 128, 3)
        image = io.imread(self.input_path + d.ProbeFileName)

        au_name = d.ProbeFileName.split('.')[-2].split('/')[-1]

        extract_all_patches(image, window_shape, self.stride, num_of_patches, self.rotations, self.output_path,
                            au_name, rep_num, self.mode)

    def extract_patches(self):
        all_refs = get_ref_df()


        create_dirs(self.output_path)


        window_shape = (128, 128, 3)
        rep_num = 0

        images_checked = {}
        for _, d in all_refs.iterrows():
            if d.ProbeFileID in images_checked:
                continue
            else:
                images_checked[d.ProbeFileID] = 1
            if d.IsTarget == 'Y':
                try:
                    image = io.imread(self.input_path + d.ProbeFileName)
                    mask = io.imread(self.input_path + d.ProbeMaskFileName)
                    rep_num += 1
                    image, mask = check_and_reshape(image, mask)

                    im_name = d.ProbeFileName.split('.')[-2].split('/')[-1]


                    tampered_patches, num_of_patches = find_tampered_patches(image, im_name, mask,
                                                                             window_shape, self.stride, 'nc16',
                                                                             self.patches_per_image)

                    save_patches(tampered_patches, num_of_patches, self.mode, self.rotations, self.output_path, im_name,
                                 rep_num, patch_type='tampered')
                except IOError as e:
                    rep_num -= 1
                    print(str(e))
                except IndexError:
                    rep_num -= 1
                    print('Mask and image have not the same dimensions')
            else:
                self.extract_authentic_patches(d, self.patches_per_image, rep_num)
