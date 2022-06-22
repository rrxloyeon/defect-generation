import matplotlib.pyplot as plt
import numpy as np

RED = 0
GREEN = 1
BLUE = 2

CLASS_ORANGE = 'orange'
CLASS_GREEN = 'green'

threshold = [(1500, 0.2), (700, 0.4), (550, 0.4)] # R, G, B

def guess_background (img):
    var = [np.var(img[:, :, RED]), np.var(img[:, :, BLUE]), np.var(img[:, :, GREEN])]
    for i, v in enumerate(var) :
        var[i] = -1*threshold[i][1] if v < threshold[i][0] else 1*threshold[i][1]
    if sum(var) > 0 :
        return CLASS_GREEN
    elif sum(var) <= 0 :
        return CLASS_ORANGE