from __future__ import print_function
import sys
import numpy as np
import pandas as pd
from scipy.integrate import simps
from numpy import trapz


f=open(sys.argv[1],'r')

def readfile(f):
	#data = pd.read_csv(f,header=None,delim_whitespace=True)
	y = np.loadtxt(f)
	print (y)
	#y = npdata[:,1:]
	'''
	for lines in f.readlines():
		y.append(lines.strip())
	#	print (y)
	'''
	y=np.array(y)
	
	y = y.astype('float64') 

	# Compute the area using the composite trapezoidal rule.
	area = trapz(y)
	print("area =", area)

	# Compute the area using the composite Simpson's rule.
	area = simps(y)
	print("area =", area)
	
	
	def autocorr(y, t=1):
		print (np.corrcoef(np.array([y[:-t], y[t:]])))
	
	autocorr(y)	

readfile(f)

