from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class CrosswalkDataset(CustomDataset):
    CLASSES = ('Background', 'Crosswalk', 'Roadway', 'Sidewalk')
    PALETTE = [[0, 0, 0], [255, 255, 255], [128, 0, 0], [0, 128, 0]]
    
    def __init__(self, **kwargs):
        super(CrosswalkDataset, self).__init__(img_suffix='.jpg', seg_map_suffix='.png', **kwargs)