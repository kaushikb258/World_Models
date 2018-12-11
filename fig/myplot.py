import numpy as np
import matplotlib.pyplot as plt

d1 = np.loadtxt('c1/performance_es.txt')
d2 = np.loadtxt('c2/performance_es.txt')


d3 = []
for i in range(d1.shape[0]):
   d3.append(d1[i])
for i in range(d2.shape[0]):
   d3.append(d2[i])

d3 = np.array(d3)

print(d3.shape)

plt.plot(d3[:,0],d3[:,1],label='min')
plt.plot(d3[:,0],d3[:,2],label='max')
plt.plot(d3[:,0],d3[:,3],label='mean')
plt.legend(fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('generation',fontsize=15)
plt.ylabel('avg. reward',fontsize=15)
plt.savefig('WMplot.png')
