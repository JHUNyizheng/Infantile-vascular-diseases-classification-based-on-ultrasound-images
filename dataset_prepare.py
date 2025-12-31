import os
import shutil
import time

dataset_dir = '/data/Hand_dataset/'
save_dir = '/data/Hand_dataset/dataset/'

for path in ['train', 'val']:
    picture_path = os.path.join(dataset_dir, 'hand/images/' + path)
    picture_lists = os.listdir(picture_path)
    for picture in picture_lists:
        new_name = os.path.join(save_dir, 'labels/' + path + '/segm', picture)
        old_name = os.path.join('/data/Hand_dataset/VOC2007/SegmentationClass', picture[:-4] + '.png')
        shutil.copyfile(old_name, new_name)
        print('OK')

