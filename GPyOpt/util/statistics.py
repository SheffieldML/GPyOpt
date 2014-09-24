import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

'''
Support functions for the package GPyOpt
Included here functions related to the extraction of sufficient statistics from the data
'''
def ellipse(points, nstd=2, Nb=100):
	def eigsorted(cov):
		vals, vecs = np.linalg.eigh(cov)
		order = vals.argsort()[::-1]
		order = vals.argsort()[::-1]	
		return vals[order], vecs[:,order]

	pos = points.mean(axis=0)
	cov = np.cov(points, rowvar=False)
#	cov=[[10, 8],[8, 10]]
	vals, vecs = eigsorted(cov)
	theta = np.radians(np.degrees(np.arctan2(*vecs[:,0][::-1])))
	width, height =  nstd * np.sqrt(vals)
	grid = np.linspace(0,2*np.pi,Nb)
	X= width * np.cos(grid)* np.cos(theta) - np.sin(theta) * height * np.sin(grid) + pos[0]
	Y= width * np.cos(grid)* np.sin(theta) + np.cos(theta) * height * np.sin(grid) + pos[1]
	return X,Y




