import matplotlib.pyplot as plt
import cPickle as pickle
from scipy.misc import imresize, imread

infile ='/home/dbuscombe/github_clones/sandbar_seg/example_timeseries_regis/RC0307Rf_20091012_0942_reg.jpg_out.p'

origfile = infile.split('_out.p')[0]

with open(infile, 'rb') as f:                                                                                    
   dat = pickle.load(f)

img = imresize(imread(origfile),dat['scale'])


plt.imshow(img) #dat['out_img'])
plt.plot(dat['contour_x'], dat['contour_y'],'k', lw=2)
plt.plot(dat['rect_x'], dat['rect_y'], 'y')
plt.plot(dat['bg_x'], dat['bg_y'], 'r')
plt.plot(dat['fg_x'], dat['fg_y'], 'w')
plt.axis('tight'); plt.show()
