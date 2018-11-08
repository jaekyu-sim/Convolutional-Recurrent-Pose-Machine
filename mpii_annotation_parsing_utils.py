#
# parsing_utils.py
# =============================================================================
"""mpii annotation parsing utils"""


def get_data_dict(value_dict):
  """Make json line data

  Returns:
  """
  data_bone = {'file_name': None,
               'is_train': None,
               'scale': None,
               'parts': None,
               'visibility': None,
               'num_parts': None,
               'obj_pos': None}

  for key in value_dict:
    if key in data_bone:
      data_bone[key] = value_dict[key]
    else:
      assert False, "No Key - '{}'".format(key)

  return data_bone

def get_parts(points):
  """Return. {id:(col, row)} dictionaries.

  Args:
    points:

  Returns:
    parts: dictionary with parts 'id' as key
    visibility: ~
  """
  parts = {str(p_id[0, 0]): (int(p_row[0, 0]), int(p_col[0, 0]))
           for p_id, p_col, p_row
           in zip(points['id'][0], points['x'][0], points['y'][0])}
  visibility = {}
  if 'is_visible' not in str(points.dtype):
    visibility = None
  else:
    for is_visible, p_id in zip(points['is_visible'][0], points['id'][0]):
      if is_visible:
        visibility[str(p_id[0, 0])] = 1
      else:
        visibility[str(p_id[0, 0])] = 0

  return parts, visibility

# def get_scale(annorect):
#   """Treat and return scale.

#   Args:
#     annorect:

#   Returns:
#     scale: float
#   """
#   if 'scale' in str(annorect.dtype):
#     try:
#       scale = float(np.max(annorect['scale'][0, 0]))
#     except ValueError:
#       scale = None
#   else:
#     scale = None

#   return scale
