'''
    domaincoloring using c++ implementation in multithreaded mode
    8 times faster than numba, 110 faster than python
'''

from DomainColoring import DomainColoring

import matplotlib.pyplot as plt
from timeit import default_timer as time

mf=2
w, h = 1920*mf, 1080*mf

lap=time()

dc = DomainColoring(w, h)

print('domain coloring', dc.w, 'x', dc.h, 'image, generation time:', time()-lap, 'secs')

plt.imshow( dc.get_image_np() )
plt.show()
