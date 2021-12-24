from ..builder import PIPELINES
import albumentations as A
import numpy as np

@PIPELINES.register_module()
class AffineTransform(object):
    #Affine (scale=None, translate_percent=None, translate_px=None, 
    # rotate=None, shear=None, interpolation=1, mask_interpolation=0, 
    # cval=0, cval_mask=0, mode=0, fit_output=False, always_apply=False, p=0.5)
    def __init__(self, scale=None, translate_percent=None, translate_px=None,
                rotate=None, shear=None, interpolation=1, mask_interpolation=0,
                cval=0, cval_mask=0, mode=0, fit_output=False, always_apply=False,
                p=0.5, threshold=0.2):
        self.scale = scale
        self.translate_percent = translate_percent
        self.translate_px = translate_px
        self.rotate = rotate
        self.shear = shear
        self.interpolation = interpolation
        self.mask_interpolation= mask_interpolation
        self.cval = cval
        self.cval_mask = cval_mask
        self.mode = mode
        self.fit_output = fit_output
        self.always_apply = always_apply
        self.p = p
        self.threshold = threshold


    def __call__(self, results):

        img = results['img']
        mask = results['gt_semantic_seg']

        gt_classes, counts = np.unique(mask, return_counts=True)
        gt_class_count_dict = dict(zip(gt_classes, counts))

        print(gt_class_count_dict)

        get_cw_prop = gt_class_count_dict[1] / sum(gt_class_count_dict.values())

        if self.threshold <= get_cw_prop:

            transform = A.Affine(self.scale, self.translate_percent, self.translate_px, self.rotate,
                                self.shear, self.interpolation, self.mask_interpolation, self.cval,
                                self.cval_mask, self.mode, self.fit_output, self.always_apply, self.p)(image=img, mask=mask)

            results['img'] = transform['image']
            results['gt_semantic_seg'] = transform['mask']

        return results