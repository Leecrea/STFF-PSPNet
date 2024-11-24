
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class Our_Dataset(BaseSegDataset):
    # 类别和对应的 RGB配色
    METAINFO = {
        # 'classes':['', ''],
        # 'palette':[[], []]

        
        # 'classes':['background', 'red', 'green', 'white', 'seed-black', 'seed-white'],
        # 'palette':[[127,127,127], [200,0,0], [0,200,0], [144,238,144], [30,30,30], [251,189,8]]

        #-----------build-----------------
        # 'classes':['background', 'build'],
        # 'palette':[[0, 0, 0], [128,0,0]]

        # 'classes':['1', '2', '3', '4', '5', '6', '7'],
        # 'palette':[[239,228,175], [128,0,0], [228,175,239], [0,128,0], [229,140,113], [113,229,140], [255,126,5]]

         # -----------loveda--------------
         # 'classes':['background', 'building', 'road', 'water', 'barren', 'forest', 'agriculture'],
         # 'palette':[[239,228,175], [128,0,0], [228,175,239], [0,128,0], [229,140,113], [113,229,140], [255,126,5]]

        # -----------landcover_ai_4----------
         # 'classes':['background', 'buildings', 'woodland', 'water'],
         # 'palette':[[0, 0, 0], [128, 0, 0], [0, 128, 0], [0, 0, 128]]

        #-----------landcover_ai_5-----------
        # 'classes':['background', 'buildings', 'woodland', 'water', 'road'],
        # 'palette':[[0, 0, 0], [128, 0, 0], [0, 128, 0], [0, 0, 128], [0, 128, 128]]

        #-----------yueyang-------------------
        # 'classes':['cultivated', 'woodland', 'grassland','water', 'build'],
        # 'palette':[[128, 128, 0], [0, 128, 0], [0, 128, 128], [0, 0, 128], [128, 0, 0]]

        # 'classes':['building', 'water', 'grassland','farmland', 'woodland'],
        # 'palette':[[128, 0, 0], [0, 0, 128], [0, 128, 128], [128, 0, 128], [128, 128, 0]]

        'classes':['buildings', 'woodland', 'water', 'road'],
        'palette':[[128, 0, 0], [0, 128, 0], [0, 0, 128], [0, 128, 128]]
    }
    
    # 指定图像扩展名、标注扩展名
    def __init__(self,
                 seg_map_suffix='.png',   # 标注mask图像的格式
                 reduce_zero_label=True, # 类别ID为0的类别是否需要除去
                 **kwargs) -> None:
        super().__init__(
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)