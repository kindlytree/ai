# bbox_ious

```
import torch

def bbox_ious(boxes1, boxes2):
    """ Compute IOU between all boxes from ``boxes1`` with all boxes from ``boxes2``.

    Args:
        boxes1 (torch.Tensor): List of bounding boxes
        boxes2 (torch.Tensor): List of bounding boxes

    Note:
        List format: [[xc, yc, w, h],...]
    """
    smooth = 0.05
    b1x1, b1y1 = (boxes1[:, :2] - (boxes1[:, 2:4] / 2)).split(1, 1)
    b1x2, b1y2 = (boxes1[:, :2] + (boxes1[:, 2:4] / 2)).split(1, 1)
    b2x1, b2y1 = (boxes2[:, :2] - (boxes2[:, 2:4] / 2)).split(1, 1)
    b2x2, b2y2 = (boxes2[:, :2] + (boxes2[:, 2:4] / 2)).split(1, 1)

    dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp(min=0)
    dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp(min=0)
    intersections = dx * dy

    areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    unions = (areas1 + areas2.t()) - intersections + smooth

    return intersections / unions

def bbox_iou(box1, box2):

    b1x1, b1y1 = box1[:2].split(1)
    b1x2, b1y2 = box1[2:4].split(1)
    b2x1, b2y1 = (box2[:2] - box2[2:4] / 2).split(1)
    b2x2, b2y2 = (box2[:2] + box2[2:4] / 2).split(1)

    dx = (b1x2.min(b2x2) - b1x1.max(b2x1)).clamp(min=0)
    dy = (b1y2.min(b2y2) - b1y1.max(b2y1)).clamp(min=0)
    intersections = dx * dy

    area1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    area2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    unions = (area1 + area2) - intersections

    return  intersections / unions

```