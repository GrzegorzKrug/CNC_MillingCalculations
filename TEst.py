
import numpy as np


def arc(angle, radius):
    return 2 * np.pi * radius * angle / 360


for ang in [15, 45, 90, 180]:
    print(f"{ang}, {arc(ang, 2):>4.3f}")
