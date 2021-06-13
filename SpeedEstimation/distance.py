import math
import numpy as np
import pandas as pd

def calculate(npy_values, time_diff):
    # todo
    h = 8 # 8m
    # angle = 0
    # middle = h / math.cos(angle)
    middle_ratio = pd.DataFrame(np.load('../output/210612204000/0_204000_origin_disp.npy'))[1170][1086]
    middle = 300 # 300m

    depth_frame1 = npy_values[0] * middle / middle_ratio
    depth_frame2 = npy_values[1] * middle / middle_ratio

    horizontal_frame1 = math.sqrt(depth_frame1**2 - h**2)
    horizontal_frame2 = math.sqrt(depth_frame2**2 - h**2)

    speed = (horizontal_frame1 - horizontal_frame2) / time_diff
    speed = speed / 1000 * 3600

    return round(speed, 1)
