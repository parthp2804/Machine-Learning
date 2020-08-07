### Creating fake data
from sklearn.datasets.samples_generator import make_blobs
X,y = make_blobs(n_samples=200, centers=2, n_features=2, cluster_std=5, random_state=11)
m = 200

### sigmoid function
import numpy as np 
def sigmoid(z):
	return 1 / (1+np.exp(-z))

### hypothesis
def h(w,X):
	z = np.array(w[0] + w[1]*np.array(X[:,0]) + w[2]*np.array(X[:,1]))
	return sigmoid(z)

### cost function (Maximum Likelihood Estimate)
def cost(w,X,y):
	return -1* np.sum(y*np.log(h(w,X)) + (1-y)*np.log(1-h(w,X)))

### partial derivative
def grad(w,X,y):
	g = [0]*3
	g[0] =  -1* np.sum(y*(1-h(w,X)) - (1-y)*h(w,X))
	g[1] =  -1* np.sum(y*(1-h(w,X))*X[:,0] - (1-y)*h(w,X)*X[:,0])
	g[2] =  -1* np.sum(y*(1-h(w,X))*X[:,1] - (1-y)*h(w,X)*X[:,1])
	return g

### gradient descent 
def descent(w_new,w_prev,lr):
	print(w_prev)
	print(cost(w_prev,X,y))
	j=0
	while True:
		w_prev = w_new
		w0 = w_prev[0] - lr*grad(w_prev,X,y)[0]
		w1 = w_prev[1] - lr*grad(w_prev,X,y)[1]
		w2 = w_prev[2] - lr*grad(w_prev,X,y)[2]
		w_new = [w0,w1,w2]
		print(w_new)
		print(cost(w_new,X,y))
		if ((w_new[0] - w_prev[0])**2 + (w_new[1] - w_prev[1])**2 + (w_new[2] - w_prev[2])**2) <pow(10,-6):
			return w_new
		if j>100:
			return w_new
		j+=1

w = [1,1,1]

### Train the model
w = descent(w,w,.0099)
print(w)

