#from numpy import linalg as LA
#
#a = [1, 1, 2]
#b = [1, 1, 3]
#
#print(LA.norm(a))

from scipy.spatial import distance
a = [1, 1, 3]
b = [1, 1, 6]
dst = distance.euclidean(a, b)
print(dst)
