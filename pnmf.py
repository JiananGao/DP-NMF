#Author: Satwik Bhattamishra

import numpy as np
import numpy.linalg as LA
from sys import exit


class PNMF():

	"""

	Attributes
	----------
	W : matrix of basis vectors
	H : matrix of coefficients
	frob_error : frobenius norm

	"""

	def __init__(self, X, W=None, H=None, rank=10, **kwargs):
		
		self.X = X       
		self._rank = rank             
	  
		
		self.X_dim, self._samples = self.X.shape

		if W is None:
			self.initialize_w()
		else:
			self.W = W

		if H is None:
			self.initialize_h()
		else:
			self.H = H
		
	def frobenius_norm(self):
		""" Euclidean error between X and W*H """

		if hasattr(self,'H') and hasattr(self,'W'):
			error = LA.norm(self.X - np.dot(self.W, self.H))            
		else:
			error = None

		return error
		
	def initialize_w(self):
		""" Initalize W to random values [0,1]."""

		self.W = np.random.random((self.X_dim, self._rank)) 
		
	def initialize_h(self):
		""" Initalize H to random values [0,1]."""

		self.H = np.random.random((self._rank, self._samples)) 


	def check_non_negativity(self):

		if self.X.min()<0:
			return 0
		else:
			return 1
	
	def compute_factors(self, max_iter=100, alpha= 0.2, beta= 0.2):
	
		if self.check_non_negativity():
			pass
		else:
			print "The given matrix contains negative values"
			exit()

		if not hasattr(self,'W'):
			self.initialize_w()
			   
		if not hasattr(self,'H'):
			self.initialize_h()

		self.frob_error = np.zeros(max_iter)

		for i in xrange(max_iter):

			self.update_h(alpha)  
			self.update_w(beta)                                      
		 
			self.frob_error[i] = self.frobenius_norm()   


	def update_h(self, beta):

		XtW = np.dot(self.W.T, self.X)
		HWtW = np.dot(self.W.T.dot(self.W), self.H ) + beta+ 2**-8
		self.H *= XtW
		self.H /= HWtW



	def update_w(self, alpha):

		XH = self.X.dot(self.H.T)
		WHtH = self.W.dot(self.H.dot(self.H.T)) + alpha+ 2**-8
		self.W *= XH
		self.W /= WHtH


