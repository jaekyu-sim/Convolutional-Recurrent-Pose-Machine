#
# mpii_annotation_parser.py
# =============================================================================
"""mpii annotation parser"""

import json
import sys

from scipy.io import loadmat

# pylint: disable=wrong-import-position
sys.path.append("../")
import config
del sys
import mpii_annotation_parsing_utils as utils
# pylint: enable=wrong-import-position

annot_mat_path = config.mpii_annot_path
annot_json_path = config.mpii_annot_json_path
annot_mat = loadmat(annot_mat_path)['RELEASE']

annot_annolist = annot_mat['annolist'][0, 0][0]
annot_img_train = annot_mat['img_train'][0, 0][0]
annot_single_person = annot_mat['single_person'][0, 0]
annot_act = annot_mat['act'][0, 0]
annot_video_list = annot_mat['video_list'][0, 0][0]

f = open(annot_json_path, 'a')
for annot, train_flag in zip(annot_annolist, annot_img_train):
  if train_flag:
    # ====================== Train Set ======================
    # annot.dtype.names -> ('image', 'annorect', 'frame_sec', 'vididx')
    name = annot['image']['name'][0, 0][0]
    annorect = annot['annorect']

    # annorect.dtype.names -> ('x1', 'y1', 'x2', 'y2', 'annopoints', 'scale', 'objpos')
    if 'scale' in str(annorect.dtype):
      scales = annorect['scale']
      objs = annorect['objpos']
      annopoints = annorect['annopoints']

      for scale, objpos, annopoint in zip(scales[0], objs[0], annopoints[0]):
        is_negative = False
        if not scale:
          value_dict = {'file_name': name, 'is_train': 1}
          data = utils.get_data_dict(value_dict)
          json.dump(data, f)
          f.write("\n")
        else:
          point = annopoint['point'][0, 0]

          scale = scale[0, 0]
          obj_col = objpos['x'][0, 0][0, 0]
          obj_row = objpos['y'][0, 0][0, 0]

          parts, p_is_visibles = utils.get_parts(point)

          value_dict = {'file_name': name,
                        'is_train': 1,
                        'scale': float(scale),
                        'parts': parts,
                        'visibility': p_is_visibles,
                        'num_parts': len(parts),
                        'obj_pos': (int(obj_row), int(obj_col))}
          data = utils.get_data_dict(value_dict)
          json.dump(data, f)
          f.write("\n")

    # annorect.dtype.names -> None
    else:
      value_dict = {'file_name': name, 'is_train': 1}
      data = utils.get_data_dict(value_dict)
      json.dump(data, f)
      f.write("\n")

    continue

  # ====================== Test Set ======================
  # annot.dtype.names -> ('image', 'annorect', 'frame_sec', 'vididx')
  test_name = annot['image']['name'][0, 0][0]
  test_annorect = annot['annorect']

  # check scale/objpos key
  # test_annorect.dtype.names -> ('scale', 'objpos')
  if 'scale' in str(test_annorect.dtype):
    test_scales = test_annorect['scale']
    test_objs = test_annorect['objpos']
    for scale, objpos in zip(test_scales[0], test_objs[0]):

      if not scale:
        value_dict = {'file_name': test_name, 'is_train': 0}
        data = utils.get_data_dict(value_dict)
        json.dump(data, f)
        f.write("\n")
      else:
        scale = scale[0, 0]
        obj_col = objpos['x'][0, 0][0, 0]
        obj_row = objpos['y'][0, 0][0, 0]
        value_dict = {'file_name': test_name,
                      'is_train': 0,
                      'scale': float(scale),
                      'obj_pos': (int(obj_row), int(obj_col))}
        data = utils.get_data_dict(value_dict)
        json.dump(data, f)
        f.write("\n")

  # annorect.dtype.names -> None
  else:
    value_dict = {'file_name': test_name, 'is_train': 0}
    data = utils.get_data_dict(value_dict)
    json.dump(data, f)
    f.write("\n")

f.close()
