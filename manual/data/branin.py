
from GPyOpt.fmodels.experiments2d import branin as branin_creator
import numpy as np

f = branin_creator()

def branin(x,y):
    return f.f(np.hstack([x,y]))
