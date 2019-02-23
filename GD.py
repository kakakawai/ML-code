import numpy as np

X = np.array([[1,2,3],[4,5,6],[7,8,9]])
m = [0,0.1,0.5,2]

print X
print X.T
a = np.dot(X,m[1:]) + m[0]
print a