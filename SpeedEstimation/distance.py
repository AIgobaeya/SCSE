import math

def calculate(npy_values, fps):
    # todo
    h = 8 # 8m
    # angle = 0
    # middle = h / math.cos(angle)
    middle_ratio = min(npy_values[0][0], npy_values[1][0])
    middle = 100

    depth_frame1 = npy_values[0][1] * middle / middle_ratio
    depth_frame2 = npy_values[1][1] * middle / middle_ratio

    horizontal_frame1 = math.sqrt(depth_frame1**2 - h**2)
    horizontal_frame2 = math.sqrt(depth_frame2**2 - h**2)

    speed = (horizontal_frame1 - horizontal_frame2) / (30 / fps)
    speed = speed / 1000 * 3600

    return round(speed, 1)