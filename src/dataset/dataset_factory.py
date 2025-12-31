from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .datasets.Disease import Disease
from .datasets.Disease_box import DiseaseBox
import torch

dataset_factory = {
  'disease': Disease,
  'diseasebox': DiseaseBox
}


def my_collate(batch):
  image_gray, image_color, gray_coords, phlebolith_label, classes = zip(*batch)
  image_gray = torch.stack([b for b in image_gray], 0)
  image_color = torch.stack([b for b in image_color], 0)
  gray_coords = torch.stack([torch.from_numpy(b) for b in gray_coords], 0)
  classes = torch.stack([torch.Tensor([b]) for b in classes], 0)

  phlebolith_label_out = {}
  all_img = [b['im_file'].split('/')[-1] for b in phlebolith_label]

  batch_idx = []
  for b in phlebolith_label:
    if len(b['batch_idx']) != 0:
      for k in range(len(b['batch_idx'])):
        batch_idx.append(all_img.index(b['batch_idx'][0]))

  phlebolith_label_out.update(
    {
      'im_file': [b['im_file'] for b in phlebolith_label],
      'ori_shape': [b['ori_shape'] for b in phlebolith_label],
      'resized_shape': [b['resized_shape'] for b in phlebolith_label],
      'ratio_pad': [b['ratio_pad'] for b in phlebolith_label],
      'img': torch.stack([b['img'] for b in phlebolith_label], 0),
      'cls': torch.concat([torch.Tensor(b['cls']) for b in phlebolith_label], 0).view(-1, 1),
      'bboxes': torch.concat([torch.from_numpy(b['bboxes']) for b in phlebolith_label], 0).view(-1, 4),
      # 'batch_idx': torch.stack([torch.Tensor([all_img.index(b['batch_idx'][0])]) for b in phlebolith_label
      #                           for k in range(len(b['batch_idx']))], 0).view(-1),
      'batch_idx': torch.concat([torch.Tensor([batch_idx])], 0).view(-1),
    }
  )
  return image_gray, image_color, gray_coords, phlebolith_label_out, classes

def get_dataset(dataset):
  return dataset_factory[dataset]

