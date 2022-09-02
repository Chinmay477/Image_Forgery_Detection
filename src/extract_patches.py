from src.patch_extraction.patch_extractor_casia import PatchExtractorCASIA

pe = PatchExtractorCASIA(input_path='../data/CASIA2', output_path='patches',
                         patches_per_image=2, stride=128, rotations=4, mode='rot')
pe.extract_patches()

