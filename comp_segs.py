import cPickle as pickle
from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np

# Use Green's theorem to compute the area
# enclosed by the given contour.
def area(x,y):
    a = 0
    x0 = x[0]
    y0 = y[0]
    for x1, y1 in zip(x[1:], y[1:]):
        dx = x1-x0
        dy = y1-y0
        a += 0.5*(y0*dx - x0*dy)
        x0 = x1
        y0 = y1
    return np.abs(a)


direc1 = '/run/media/dbuscombe/MASTER/GCMRC/SANDBAR_REMOTECAMERAS/RC0307Rf_segres/'
filenames1 = sorted(glob(direc1+'*.p'))

direc2 = '/run/media/dbuscombe/MASTER/GCMRC/SANDBAR_REMOTECAMERAS/RC0307Rf_segres/caleb_res/'
filenames2 = sorted(glob(direc2+'*.p'))

files1 = []
for filename in filenames1:
    files1.append(filename.split(os.sep)[-1])

files2 = []
for filename in filenames2:
    files2.append(filename.split(os.sep)[-1])

files_use = list(set(files1) & set(files2))

A1 = []; A2 = []
for this_file in files_use:
   dat1 =  pickle.load( open( direc1+this_file, "rb" ) )
   dat2 =  pickle.load( open( direc2+this_file, "rb" ) )

   x = dat1['contour_x']; y = dat1['contour_y']
   A1.append(area(x,y))

   x = dat2['contour_x']; y = dat2['contour_y']
   A2.append(area(x,y))

   fig = plt.figure()
   plt.imshow(dat1['img'])
   plt.plot(dat1['contour_x'], dat1['contour_y'],'b')
   plt.plot(dat2['contour_x'], dat2['contour_y'],'r')
   plt.axis('tight')
   plt.axis('off')
   plt.savefig(this_file+".png")
   plt.close()
   del fig

plt.plot(100*((np.asarray(A1)-np.asarray(A2))/np.asarray(A1)),'r')
plt.ylabel('Percentage difference in area (pixels)'); plt.xlabel('Image Number')
plt.savefig('areal_discrep_dan_caleb_rm30_44images.png'); plt.close()
