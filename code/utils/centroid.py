def get_centroid(x1, y1, x2, y2):
    """
    Compute centroid of a bounding box.
    (x1, y1) -> top-left
    (x2, y2) -> bottom-right
    """
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return cx, cy
