
from fnmatch import filter 
import cPickle as pickle
import numpy.ma as ma
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyproj
from scipy.misc import imread, imresize
import scipy.misc

import warnings
warnings.filterwarnings("ignore")


def nancrop(a, x, y):
   nans = a[:,:,0]==0 #np.isnan(a)
   nancols = np.all(nans, axis=0) # 
   nanrows = np.all(nans, axis=1) # 

   firstcol = nancols.argmin() # 
   firstrow = nanrows.argmin() #

   lastcol = len(nancols) - nancols[::-1].argmin() # 
   lastrow = len(nanrows) - nanrows[::-1].argmin() #

   return a[firstrow:lastrow,firstcol:lastcol,:], x[firstrow:lastrow], y[firstcol:lastcol]


def dotrans(img,trans,size,mappts):

   dst = cv2.warpPerspective(img,trans,size)

   pts2 = np.vstack((mappts[:,0],mappts[:,1])).T

   # find minimum in e,n
   minpts2 = np.min(pts2, axis=0)

   rows,cols,ch = dst.shape

   N_axis = np.linspace(minpts2[1], minpts2[1]+rows, rows)
   E_axis = np.linspace(minpts2[0], minpts2[0]+cols, cols)

   return N_axis, E_axis, dst


root = '/home/dbuscombe/github_clones/sandbar_seg/'

cs2cs_args = "epsg:26949"
trans =  pyproj.Proj(init=cs2cs_args)

#coords for 30mile
lon = -111.8476
lat = 36.5159
campos = trans(lon, lat)
campos = campos[1], campos[0]

bkimg = imread(root+'tifs/RM30_7.tif')
x,y,d = bkimg.shape
pos = np.genfromtxt(root+'tifs/RM30_7.tfw')
imextent = [ pos[4], pos[4]+y*pos[0], pos[5]-x*pos[0], pos[5] ]

dat =  pickle.load( open( "RC0307Rf_homo_mergedtrans.p", "rb" ) )

T = dat['trans']

infile ='/home/dbuscombe/github_clones/sandbar_seg/example_timeseries_regis/RC0307Rf_20091012_0942_reg.jpg_out.p'

origfile = infile.split('_out.p')[0]

with open(infile, 'rb') as f:                                                                                    
   segdat = pickle.load(f)

#img = imresize(imread(origfile),dat['scale'])

img = imread(origfile)

fct = 1/segdat['scale']


plt.imshow(img) #dat['out_img'])
plt.plot(fct*segdat['contour_x'], fct*segdat['contour_y'],'r', lw=2)
plt.axis('tight'); plt.show()


x,y,d = img.shape
#img = img[500:1500,:y,:d]
N_axis, E_axis, dst = dotrans(img,T,dat['dsize'],dat['mappts'])

rectim, N, E = nancrop(dst, N_axis, E_axis)


#pts = np.vstack((fct*segdat['contour_x'], fct*segdat['contour_y'], np.ones(len(segdat['contour_x']))))


#Tinv = cv2.invert(T)[1]

# get the elements in the transform matrix
h0 = T[0,0]
h1 = T[0,1]
h2 = T[0,2]
h3 = T[1,0]
h4 = T[1,1]
h5 = T[1,2]
h6 = T[2,0]
h7 = T[2,1]
h8 = T[2,2]

tx = (h0*fct*segdat['contour_x'] + h1*fct*segdat['contour_y'] + h2) 
ty = (h3*fct*segdat['contour_x'] + h4*fct*segdat['contour_x'] + h5) 
tz = (h6*fct*segdat['contour_x'] + h7*fct*segdat['contour_y'] + h8)

tx = tx/tz
ty = ty/tz

plt.imshow(ma.masked_where(rectim==0,rectim), origin='lower' )
plt.plot(tx, ty/tz + fct**2, 'r')
plt.show()



#   ax = plt.subplot(111)
#   plt.imshow(bkimg, extent = imextent)
#   plt.xlim(campos[1]-75, campos[1]+75)
#   plt.ylim(campos[0]-100, campos[0]+125)

#   plt.imshow(ma.masked_where(rectim==0,rectim), extent = [E.min(), E.max(), N.min(), N.max()], origin='lower' )

#   plt.plot(tx+E.min(), ty+N.min(), 'r')

#   plt.show()


#   ax.get_xaxis().get_major_formatter().set_useOffset(False)
#   ax.get_yaxis().get_major_formatter().set_useOffset(False)
#   labels = ax.get_xticklabels()
#   plt.setp(labels, rotation=30, fontsize=10)
#   labels = ax.get_yticklabels()
#   plt.setp(labels, rotation=30, fontsize=10)













