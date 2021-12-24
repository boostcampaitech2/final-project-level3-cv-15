from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class CrosswalkDataset(CustomDataset):
    CLASSES = ('Background', 'Crosswalk')
    PALETTE = [[0, 0, 0], [255, 255, 255]]
    
    def __init__(self, **kwargs):
        super(CrosswalkDataset, self).__init__(img_suffix='.jpg', seg_map_suffix='.png', **kwargs)