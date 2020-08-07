import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import style
### random dataset and added bias for simplicity
X = np.array([
    [-2,4,-1],
	[4,1,-1],
	[1,6,-1],
	[2,4,-1],
	[6,2,-1]
	])

y = np.array([-1,-1,1,1,1])

### stochastic gradient descent 

def sgd(X,y):
	w = np.zeros(len(X[0]))
	lr = 1
	epochs = 100000

	for epoch in range(1,epochs):
		for i,x in enumerate(X):
			#print(i)
			#print(x)
			if (y[i]*np.dot(X[i],w)) < 1:
				w = w + lr*((y[i]*X[i]) + (-2 * (1/epoch)*w))
			else:
				w = w+ lr*(-2 * (1/epoch)*w)

	return w

### print the errors i.e misclassified and classified examples
def svm_err(X,y):
	w = np.zeros(len(X[0]))
	lr = 1
	epochs = 100000
	errors = []

	for epoch in range(1,epochs):
		error = 0
		for i,x in enumerate(X):
			if (y[i]*np.dot(X[i],w)) < 1:
				w = w + lr*((y[i]*X[i]) + (-2 * (1/epoch)*w))
				error = 1
			else:
				w = w+ lr*(-2 * (1/epoch)*w)
		errors.append(error)
		print('Error:', errors)


w = sgd(X,y)
svm_err(X,y)
print(w)

