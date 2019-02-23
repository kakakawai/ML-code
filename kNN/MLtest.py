import numpy as np
import operator
import kNN

import matplotlib as mp
import matplotlib.pyplot as plt
'''
rateList = []
hList = []
for i in np.arange(0.1,1,0.1):
    print 'h:%.1f' %(i)
    rate = kNN.datingClassTest(i)
    rateList.append(rate)
    hList.append(i)

print rateList
plt.plot(hList,rateList,linewidth='3',color='r',marker='o')
plt.xlabel('hoRatio')
plt.ylabel('Error Rate')
plt.show()
'''
kNN.numberClassTest()