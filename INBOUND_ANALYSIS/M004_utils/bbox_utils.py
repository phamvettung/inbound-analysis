import numpy as np

def real_velocity(delta_d, fps, scale):
    print(f"duaration {fps}")
    delta_t = fps
    velocity_in_pixels_per_second = delta_d/delta_t
    velocity_in_mili_meter_per_second = (velocity_in_pixels_per_second * scale)/1000
    return velocity_in_mili_meter_per_second

def pixel_to_meter_scale(mili_meter_ref, pixel_ref):
    return mili_meter_ref/pixel_ref

def compute_displacement_in_pixels(center_1, center_2):
    cx1, cy1 = center_1
    cx2, cy2 = center_2
    delta_d_pixel = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
    return delta_d_pixel

def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)




