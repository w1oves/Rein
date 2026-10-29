import numpy as np
import pytest
from reinpp.labels import cityscapes_lookup, mapillary_lookup, convert_mask


def test_cityscapes_label_ids_and_void_are_distinct_from_train_ids():
    raw = np.array([[7, 8, 26, 33, 0, 255]], dtype=np.uint8)
    assert convert_mask(raw, cityscapes_lookup()).tolist() == [[0, 1, 13, 18, 255, 255]]


def test_mapillary_protocol_groups_and_ignored_labels():
    names = ['Road', 'Curb Cut', 'Traffic Sign (Back)', 'On Rails', 'Bicyclist',
             'Street Light', 'Lane Marking - General', 'Ground Animal']
    labels = [{'readable': s} for s in names + ['Unlabeled'] * (66 - len(names))]
    assert mapillary_lookup(labels)[:8].tolist() == [0, 1, 7, 16, 12, 5, 0, 255]


def test_rgb_masks_and_incompatible_mapillary_version_fail():
    with pytest.raises(ValueError, match='not RGB'):
        convert_mask(np.zeros((4, 4, 3), dtype=np.uint8), cityscapes_lookup())
    with pytest.raises(ValueError, match='v1.2'):
        mapillary_lookup([{'readable':'road'}])
