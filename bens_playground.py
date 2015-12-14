from hyperspy.signals import *
import numpy as np

dat = np.array([np.arange(3000).reshape(30,100), -np.arange(3000).reshape(30,100) + 3000])
test = Image(dat)
test.plot()
