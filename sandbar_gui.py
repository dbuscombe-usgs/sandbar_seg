## has been developed at the Grand Canyon Monitoring & Research Center,
## U.S. Geological Survey
##
## Author: Daniel Buscombe
## Project homepage: <https://github.com/dbuscombe-usgs/sandbar_seg>
##
##This software is in the public domain because it contains materials that originally came from
##the United States Geological Survey, an agency of the United States Department of Interior.
##For more information, see the official USGS copyright policy at
##http://www.usgs.gov/visual-id/credit_usgs.html#copyright
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
## See the GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program. If not, see <http://www.gnu.org/licenses/>.

'''
===============================================================================
Interactive Sandbar Segmentation using GrabCut algorithm.
A Program by Daniel Buscombe, USGS
April - May 2016

README FIRST:
    Two windows will show up, one for input and one for output.

    At first, in input window, draw a rectangle around the object using
mouse right button. Then press 'n' to segment the object (once or a few times)
For any finer touch-ups, you can press any of the keys below and draw lines on
the areas you want. Then again press 'n' for updating the output.

Key '0' - To select areas of sure background
Key '1' - To select areas of sure foreground

Key 'n' - To update the segmentation
Key 'r' - To reset the setup
Key 's' - To save the results
===============================================================================
'''

from __future__ import print_function
from Tix import *

try:
    import Tkinter
    import tkFont
except ImportError: # py3k
    import tkinter as Tkinter
    import tkinter.font as tkFont

import ttk

from tkFileDialog import askopenfilename
import os
from glob import glob

from ttkcalendar import *
import calendar
import tkMessageBox
from ScrolledText import ScrolledText

from scipy.misc import imresize, imread
from scipy.signal import convolve2d
from skimage.morphology import remove_small_holes, dilation, disk
from skimage.measure import regionprops, label

import numpy as np
import cv2

import cPickle as pickle
import datetime as DT
import calendar

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as md
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# chaneg this for windows:
rootfolder = '/run/media/dbuscombe/MASTER/GCMRC/SANDBAR_REMOTECAMERAS/'

##============================================
def ani_frames(infiles):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #im = ax.imshow(imresize(imread(infiles[0]),.25))
    im = ax.imshow(imread(infiles[0]))

    def init():
       im.set_data([[]])
       return im

    def update_img(i):
        #im.set_data(imresize(imread(infiles[i]),.25))
        im.set_data(imread(infiles[i]))

        ext = os.path.splitext(infiles[i])[1][1:]
        date = infiles[i].split(os.sep)[-1].split('RC')[-1].split('_')[1]
        time = infiles[i].split(os.sep)[-1].split('RC')[-1].split('_')[2].split('.'+ext)[0]

        plt.title(DT.datetime.strptime(date+' '+time, '%Y%m%d %H%M').strftime('%d %b %Y, %H:%M'))
        #plt.title(infiles[i].split(os.sep)[-1].split('.')[0])
        return im

    ani = animation.FuncAnimation(fig,update_img, frames=len(infiles), interval=100, init_func = init, save_count=len(infiles))
    if animation.writers.is_available('ffmpeg'):
       print('using ffmpeg to compile video')
       writer = animation.writers['ffmpeg'](fps=1)
    else:
       print('using avconv to compile video')
       writer = animation.writers['avconv'](fps=1)

    ani.save(infiles[0].split(os.sep)[-1].split('.')[0]+'_'+infiles[-1].split(os.sep)[-1].split('.')[0]+'.mp4',writer=writer,dpi=600)
    del fig
    print("Done!")
    return ani

##============================================
def ani_frames_withQdat(infiles, dat):

    fig = plt.figure()
    fig.subplots_adjust(wspace = 0.2, hspace=0.2, left=0, right=1, bottom=0, top=1)

    Q = []; D = []; T = []; datenums = []
    image_times = []
    for infile in infiles:
       ext = os.path.splitext(infile)[1][1:]
       date = infile.split(os.sep)[-1].split('RC')[-1].split('_')[1]
       time = infile.split(os.sep)[-1].split('RC')[-1].split('_')[2].split('.'+ext)[0]

       image_time = toTimestamp(DT.datetime.strptime(date+' '+time, '%Y%m%d %H%M'))+ 6 * 60 * 60
       image_times.append(image_time)
       Q.append(np.interp(image_time,dat['timeunix'],dat['Qcfs']))
       D.append(date)
       T.append(time)
       datenums.append(DT.datetime.strptime(date+' '+time, '%Y%m%d %H%M'))

    ax2 = fig.add_subplot(212, aspect=.005)
    #ax2.set_aspect('auto')
    #ax2 = plt.axes(ylim=(np.min(np.asarray(Q))-100, np.max(np.asarray(Q))+100))

    xfmt = md.DateFormatter('%Y-%m-%d')
    ax2.xaxis.set_major_formatter(xfmt)

    ti = np.arange(image_times[0], image_times[-1], 60)
    Qall = np.interp(ti,dat['timeunix'],dat['Qcfs'])

    tiplot = []
    for i in xrange(len(ti)):
       tiplot.append(DT.datetime.fromtimestamp(ti[i]))

    #plt.plot(datenums,Q,'k')
    plt.plot(tiplot, Qall,'k')
    plt.subplots_adjust(bottom=0.2)
    plt.xticks( rotation=25 )
    plt.yticks( rotation=25 )
    plt.ylabel('Discharge (cfs)')

    line, = ax2.plot([], [], 'ro', lw=2)

    ax = fig.add_subplot(211)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    im = ax.imshow(imresize(imread(infiles[0]),.25), interpolation='nearest')

    def init():
       im.set_data([[]])
       line.set_data([], [])
       return im, line

    def update_img(i):
        im.set_data(imresize(imread(infiles[i]),.25))
        line.set_data(datenums[i], Q[i])
        try:
           ax.title(DT.datetime.strptime(D[i]+' '+T[i], '%Y%m%d %H%M').strftime('%d %b %Y, %H:%M')+' --  '+str(Q[i]).split('.')[0]+' cfs')
        except:
           pass
        return im, line

    ani = animation.FuncAnimation(fig,update_img, frames=len(infiles), interval=100, init_func = init, save_count=len(infiles))
    if animation.writers.is_available('ffmpeg'):
       print('using ffmpeg to compile video')
       writer = animation.writers['ffmpeg'](fps=1)
    else:
       print('using avconv to compile video')
       writer = animation.writers['avconv'](fps=1)

    ani.save(infiles[0].split(os.sep)[-1].split('.')[0]+'_'+infiles[-1].split(os.sep)[-1].split('.')[0]+'.mp4',writer=writer,dpi=600)
    del fig
    print("Done!")
    return ani

##============================================
def toTimestamp(d):
  return calendar.timegm(d.timetuple())

##======================================================
def load_gagedata(nearest_gage):
   if nearest_gage == 0:
      dat =  pickle.load( open( "rm0_time_Qcfs.p", "rb" ) )
   elif nearest_gage == 1:
      dat =  pickle.load( open( "rm30_time_Qcfs.p", "rb" ) )
   elif nearest_gage == 2:
      dat =  pickle.load( open( "rm61_time_Qcfs.p", "rb" ) )
   elif nearest_gage == 3:
      dat =  pickle.load( open( "rm87_time_Qcfs.p", "rb" ) )
   elif nearest_gage == 4:
      dat =  pickle.load( open( "rm166_time_Qcfs.p", "rb" ) )
   elif nearest_gage == 5:
      dat =  pickle.load( open( "rm225_time_Qcfs.p", "rb" ) )
   else:
      print("error specifiying gage")

   return dat
   #cfs to cms = 0.028316847

#======================================================
def read_image(filename, scale):
   img = imresize(cv2.imread(filename),scale) #resize image so quarter size
   imagehsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
   im = imresize(cv2.imread(filename,0),scale) #resize image so quarter size
   la = cv2.Laplacian(im,cv2.CV_64F)
   # get std and mean through stndard deviation, fast thru convolution
   m1, s1 = std_convoluted(im, .5)
   m2, s2 = std_convoluted(im, .25)
   return img, imagehsv, im, la, m1, s1, m2, s2

#=====================================================
def clean_mask(mask, imagehsv, s1, s2, m1, m2):
    try:
       mask[imagehsv[:,:,0]>np.percentile(imagehsv[:,:,0],75)] = 2
       mask[imagehsv[:,:,1]>np.percentile(imagehsv[:,:,1],75)] = 2
       mask[imagehsv[:,:,2]<np.percentile(imagehsv[:,:,2],50)] = 2

       mask[s1>np.percentile(s1,75)] = 2
       mask[s2>np.percentile(s2,75)] = 2
       mask[la>np.percentile(la,75)] = 2

       mask[m1<np.percentile(m1,50)] = 2
       mask[m2<np.percentile(m2,50)] = 2
    except:
       pass

    return mask

#=====================================================
def finalise_mask(mask2, Athres):
    try:
       l = label(mask2)
       for region in regionprops(l):
          if (region.area<Athres):
             l[l==region.label] = 0

       mask2 = (l>0).astype('uint8')

       mask2 = dilation(mask2, disk(3)).astype('uint8')
       mask2 = remove_small_holes(mask2, min_size=10000).astype('uint8')
    except:
       pass

    return mask2

##======================================================
def std_convoluted(image, N):
    """
    fast windowed mean and stadev based on kernel convolution
    """
    im = np.array(image, dtype=float)
    im2 = im**2
    ones = np.ones(im.shape)

    kernel = np.ones((2*N+1, 2*N+1))
    s = convolve2d(im, kernel, mode="same")
    s2 = convolve2d(im2, kernel, mode="same")
    ns = convolve2d(ones, kernel, mode="same")

    return s/ns , np.sqrt((s2 - s**2 / ns) / ns)

##======================================================
def onmouse(event,x,y,flags,param):
    global img,img2,drawing,value,mask,rectangle,rect,rect_or_mask,ix,iy,rect_over

    BLUE = [255,0,0]        # rectangle color
    BLACK = [0,0,0]         # sure BG
    WHITE = [255,255,255]   # sure FG

    DRAW_BG = {'color' : BLACK, 'val' : 0}
    DRAW_FG = {'color' : WHITE, 'val' : 1}
    thickness = 2           # brush thickness

    # Draw Rectangle
    if event == cv2.EVENT_RBUTTONDOWN:
        rectangle = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if rectangle == True:
            img = img2.copy()
            cv2.rectangle(img,(ix,iy),(x,y),BLUE,2)
            rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
            rect_or_mask = 0

    elif event == cv2.EVENT_RBUTTONUP:
        rectangle = False
        rect_over = True
        cv2.rectangle(img,(ix,iy),(x,y),BLUE,2)
        rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
        rect_or_mask = 0
        print(" Now press the key 'n' a few times until no further change \n")

    # draw touchup curves

    if event == cv2.EVENT_LBUTTONDOWN:
        if rect_over == False:
            print("first draw rectangle \n")
        else:
            drawing = True
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing == True:
            drawing = False
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)

#################################################
def gui():

    print(__doc__)

    #=======================
    # NOTE: Frame will make a top-level window if one doesn't already exist which
    # can then be accessed via the frame's master attribute
    master = Tkinter.Frame(name='sandbar segtool')

    self = master.master  # short-cut to top-level window
    master.pack()  # pack the Frame into root, defaults to side=TOP
    self.title('Sandbar Image Processing Tool')  # name the window

    # create notebook
    demoPanel = Tkinter.Frame(master, name='demo')  # create a new frame slaved to master
    demoPanel.pack()  # pack the Frame into root

    # create (notebook) demo panel
    nb = ttk.Notebook(demoPanel, name='notebook')  # create the ttk.Notebook widget

    # extend bindings to top level window allowing
    #   CTRL+TAB - cycles thru tabs
    #   SHIFT+CTRL+TAB - previous tab
    nb.enable_traversal()

    nb.pack(fill=Tkinter.BOTH, expand=Tkinter.Y, padx=2, pady=3)  # add margin

    self.datloaded = 0
    self.idealtime = 'all'

    #==============================================================
    #==============================================================
    #========START about tab

    # Populate the second pane. Note that the content doesn't really matter
    read_frame = Tkinter.Frame(nb)
    nb.add(read_frame, text='Read & Process Images')#, state='disabled')

    read_frame.configure(background='GoldenRod1')

    def hello1_alt():
       root = Tk()
       root.wm_title("Read & Process Images")
       S = Scrollbar(root)
       T = Text(root, height=40, width=60, wrap=WORD)
       S.pack(side=RIGHT, fill=Y)
       T.pack(side=LEFT, fill=Y)
       S.config(command=T.yview)
       T.config(yscrollcommand=S.set)

       T.tag_configure('bold_italics', font=('Arial', 12, 'bold', 'italic'))
       T.tag_configure('big', font=('Verdana', 20, 'bold'))
       T.tag_configure('color', foreground='#476042', font=('Tempus Sans ITC', 12, 'bold'))
       T.insert(END, __doc__)

    MSG1_btn = Tkinter.Button(read_frame, text = "Instructions", command = hello1_alt)
    MSG1_btn.grid(row=0, column=0, pady=(2,4))
    MSG1_btn.configure(background='thistle3', fg="black")

    read_frame.rowconfigure(1, weight=1)
    read_frame.columnconfigure((0,1), weight=1, uniform=1)

    #=======================
    # get image files
    self.read_btn = Tkinter.Button(read_frame, text='Get image files', underline=0,
	             command=lambda :_get_images())
    self.read_btn.grid(row=1, column=0, pady=(2,4))
    self.read_btn.configure(background='thistle3', fg="black")

    #=======================
    # make movie
    self.movie_btn = Tkinter.Button(read_frame, text='Make movie', underline=0,
	             command=lambda :_make_movie())
    self.movie_btn.grid(row=1, column=1, pady=(2,4))
    self.movie_btn.configure(background='thistle3', fg="black")

    #=======================
    # process button
    proc_btn = Tkinter.Button(read_frame, text='Process!', underline=0,
	             command=lambda :_proc(self))
    proc_btn.grid(row=2, column=0, pady=(2,4))
    proc_btn.configure(background='thistle3', fg="black")

    # ========================
    # close windows
    destroy_btn = Tkinter.Button(read_frame, text='Quit', underline=0,
	             command=lambda :_quit(self))
    destroy_btn.grid(row=2, column=1, pady=(2,4))
    destroy_btn.configure(background='thistle3', fg="black")

    #=======================
    def _quit(master):
       cv2.destroyAllWindows()
       master.destroy()

    #=======================
    def _make_movie():
       ani_frames(self.imagefiles)

    #=======================
    def _make_movie_withQdat():
       ani_frames_withQdat(self.imagefiles, self.dat)

    #=======================
    def _get_images():
        self.imagefiles = askopenfilename(filetypes = [ ("Image Files", ("*.jpg", "*.JPG", '*.jpeg')), ("TIF",('*.tif', '*.tiff')), ("PNG",('*.PNG', '*.png'))] , multiple=True)

        for k in xrange(len(self.imagefiles)):
           print('image '+str(k)+' of '+str(len(self.imagefiles)-1))
           print(self.imagefiles[k])
        self.folder = os.path.dirname(self.imagefiles[0])
        
        self.update()

    #=======================
    def _proc(self):
       global img,img2,drawing,value,mask,rectangle,rect,rect_or_mask,ix,iy,rect_over
       counter = 0
       for filename in self.imagefiles:
          print('Processing ' + filename)
          print('image '+str(counter)+' of '+str(len(self.imagefiles)-1))
          counter = counter+1

          BLUE = [255,0,0]        # rectangle color
          BLACK = [0,0,0]         # sure BG
          WHITE = [255,255,255]   # sure FG
          DRAW_BG = {'color' : BLACK, 'val' : 0}
          DRAW_FG = {'color' : WHITE, 'val' : 1}
          # setting up flags
          rect = (0,0,1,1)
          drawing = False         # flag for drawing curves
          rectangle = False       # flag for drawing rect
          rect_over = False       # flag to check if rect drawn
          rect_or_mask = 100      # flag for selecting rect or mask mode
          value = DRAW_FG         # drawing initialized to FG
          thickness = 2           # brush thickness
          Athres = 1000
          scale = 0.25

          img, imagehsv, im, la, m1, s1, m2, s2 = read_image(filename, scale)
          img2 = img.copy()                               # a copy of original image
          mask = np.zeros(img.shape[:2],dtype = np.uint8) # mask initialized to PR_BG
          output = np.zeros(img.shape,np.uint8)           # output image to be shown

          # input and output windows
          cv2.namedWindow('output', cv2.WINDOW_AUTOSIZE)
          cv2.namedWindow('input', cv2.WINDOW_AUTOSIZE)
          cv2.namedWindow(filename, cv2.WINDOW_AUTOSIZE)
          cv2.setMouseCallback('input',onmouse)
          cv2.moveWindow('input',img.shape[1]+10,90)

          print(" Instructions: \n")
          print(" Draw a rectangle around the object using right mouse button \n")

          while(1):

              cv2.imshow('output',output)
              cv2.imshow(filename,img)
              cv2.imshow('input',img)
              k = 0xFF & cv2.waitKey(1)
              # key bindings
              if k == 27:         # esc to exit
                  break
              elif k == ord('0'): # BG drawing
                  print(" mark background regions with left mouse button \n")
                  value = DRAW_BG
              elif k == ord('1'): # FG drawing
                  print(" mark foreground regions with left mouse button \n")
                  value = DRAW_FG
              elif k == ord('s'): # save image
                  bar = np.zeros((img.shape[0],5,3),np.uint8)
                  #res = np.hstack((img2,bar,img,bar,output))
                  cv2.imwrite(filename+'_output.png',img) #res)
                  print(" Result saved as image \n")

                  fig=plt.figure(); cs = plt.contour(output[:,:,0]>0, [0.5], colors='r');
                  plt.close(); del fig
                  p = cs.collections[0].get_paths()[0]
                  v = p.vertices
                  x = v[:,0]
                  y = v[:,1]
                  pickle.dump( {'contour_x':x, 'contour_y':y, 'img':img2}, open( filename+"_out.p", "wb" ) )
                  #tmp = pickle.load(open('22mile.JPG_out.p', 'rb'))
                  #plt.imshow(tmp['img']); plt.plot(tmp['contour_x'], tmp['contour_y'], 'r'); plt.show()
                  break
                  cv2.destroyAllWindows()

              elif k == ord('r'): # reset everything
                  print("resetting \n")
                  rect = (0,0,1,1)
                  drawing = False
                  rectangle = False
                  rect_or_mask = 100
                  rect_over = False
                  value = DRAW_FG
                  img = img2.copy()
                  mask = np.zeros(img.shape[:2],dtype = np.uint8) # mask initialized to PR_BG
                  output = np.zeros(img.shape,np.uint8)           # output image to be shown
              elif k == ord('n'): # segment the image
                  print(""" For finer touchups, mark foreground and background after pressing keys 0-3
                  and again press 'n' \n""")
                  if (rect_or_mask == 0):         # grabcut with rect
                      bgdmodel = np.zeros((1,65),np.float64)
                      fgdmodel = np.zeros((1,65),np.float64)
                      cv2.grabCut(img2,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_RECT)

                      mask = clean_mask(mask, imagehsv, s1, s2, m1, m2)

                      rect_or_mask = 1
                  elif rect_or_mask == 1:         # grabcut with mask
                      bgdmodel = np.zeros((1,65),np.float64)
                      fgdmodel = np.zeros((1,65),np.float64)
                      cv2.grabCut(img2,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_MASK)

              mask2 = np.where((mask==1) + (mask==3),255,0).astype('uint8')
              mask2 = finalise_mask(mask2, Athres)
              output = cv2.bitwise_and(img2,img2,mask=mask2)

          cv2.destroyAllWindows()

    #==============================================================
    #==============================================================
    #========END 1st tab

    #==============================================================
    #==============================================================
    #========START 2nd tab

    # Populate the second pane. Note that the content doesn't really matter
    q_frame = Tkinter.Frame(nb)
    nb.add(q_frame, text='Process Images Based on Site/Time/Discharge')

    q_frame.configure(background='SeaGreen1')

    MSG1_btn = Tkinter.Button(q_frame, text = "Instructions", command = hello1_alt)
    MSG1_btn.grid(row=0, column=0, pady=(2,4))
    MSG1_btn.configure(background='thistle3', fg="black")

    q_frame.rowconfigure(1, weight=1)
    q_frame.columnconfigure((0,1), weight=1, uniform=1)

    #=======================
    # discharge
    self.Nvar = Tkinter.DoubleVar()
    Nscale = Tkinter.Scale( q_frame, variable = self.Nvar, from_=5000, to=45000, resolution=1000, tickinterval=1000, label = 'Discharge' )
    Nscale.set(8000)
    Nscale.grid(row=0, column=1,  pady=(2,4))
    Nscale.configure(background='thistle3', fg="black")

    #=======================
    #menu for site
    self.bb=  Tkinter.Menubutton ( q_frame, text="Choose Site", relief=Tkinter.RAISED)
    self.bb.grid(column = 0, row = 1, pady=(2,4))
    self.bb.menu  =  Tkinter.Menu ( self.bb, tearoff = 1 , background='PaleVioletRed1', fg="black")

    sitelist = np.genfromtxt('sites.txt', dtype=str)

    submenu1 = Menu(self.bb.menu)
    for site in xrange(11):
       submenu1.add_command(label=sitelist[site], command = lambda v=site: _SetSitePick(master, v),  font=('Arial', 10, 'bold', 'italic'))

    self.bb.menu.add_cascade(label='Upper Marble Canyon', menu=submenu1, underline=0)

    submenu2 = Menu(self.bb.menu)
    for site in xrange(12,34):
       submenu2.add_command(label=sitelist[site], command = lambda v=site: _SetSitePick(master, v),  font=('Arial', 10, 'bold', 'italic'))

    self.bb.menu.add_cascade(label='Lower Marble Canyon', menu=submenu2, underline=0)

    submenu3 = Menu(self.bb.menu)
    for site in xrange(35, 47):
       submenu3.add_command(label=sitelist[site], command = lambda v=site: _SetSitePick(master, v),  font=('Arial', 10, 'bold', 'italic'))

    self.bb.menu.add_cascade(label='Eastern Grand Canyon', menu=submenu3, underline=0)

    submenu4 = Menu(self.bb.menu)
    for site in xrange(48,71):
       submenu4.add_command(label=sitelist[site], command = lambda v=site: _SetSitePick(master, v),  font=('Arial', 10, 'bold', 'italic'))

    self.bb.menu.add_cascade(label='Western Grand Canyon', menu=submenu4, underline=0)

    self.bb["menu"]  =  self.bb.menu
    self.bb.configure(background='thistle3', fg="black")

    #=======================
    #menu for time
    self.Mb=  Tkinter.Menubutton ( q_frame, text="Set time", relief=Tkinter.RAISED)
    self.Mb.grid(column = 1, row = 1, pady=(2,4))
    self.Mb.menu  =  Tkinter.Menu ( self.Mb, tearoff = 1 , background='orchid2', fg="black" )
    self.Mb.menu.add_command(label="All times", command = lambda v=1: _SetTime(v))
    self.Mb.menu.add_command(label="Ideal time for site", command = lambda v=2: _SetTime(v))
    self.Mb.menu.add_command(label="06:00 -- 07:00", command = lambda v=3: _SetTime(v))
    self.Mb.menu.add_command(label="07:00 -- 08:00", command = lambda v=4: _SetTime(v))
    self.Mb.menu.add_command(label="08:00 -- 09:00", command = lambda v=5: _SetTime(v))
    self.Mb.menu.add_command(label="09:00 -- 10:00", command = lambda v=6: _SetTime(v))
    self.Mb.menu.add_command(label="10:00 -- 11:00", command = lambda v=7: _SetTime(v))
    self.Mb.menu.add_command(label="11:00 -- 12:00", command = lambda v=8: _SetTime(v))
    self.Mb.menu.add_command(label="12:00 -- 13:00", command = lambda v=9: _SetTime(v))
    self.Mb.menu.add_command(label="13:00 -- 14:00", command = lambda v=10: _SetTime(v))
    self.Mb.menu.add_command(label="14:00 -- 15:00", command = lambda v=11: _SetTime(v))
    self.Mb.menu.add_command(label="15:00 -- 16:00", command = lambda v=12: _SetTime(v))
    self.Mb.menu.add_command(label="16:00 -- 17:00", command = lambda v=13: _SetTime(v))
    self.Mb.menu.add_command(label="17:00 -- 18:00", command = lambda v=14: _SetTime(v))
    self.Mb.menu.add_command(label="18:00 -- 19:00", command = lambda v=15: _SetTime(v))
    self.Mb["menu"]  =  self.Mb.menu
    self.Mb.configure(background='thistle3', fg="black")

    #=======================
    # start date button
    start = Calendar2(q_frame)
    start.grid(row=2, column=0, pady=(2,4))

    self.startdate = start.selection

    start_btn = Tkinter.Button(q_frame, text='Set Start Date', underline=0,
	             command=lambda :_qstart(start))
    start_btn.grid(row=3, column=0, pady=(2,4))
    start_btn.configure(background='thistle3', fg="black")

    #=======================
    # end date button
    end = Calendar2(q_frame)
    end.grid(row=2, column=1, pady=(2,4))

    self.enddate = end.selection

    end_btn = Tkinter.Button(q_frame, text='Set End Date', underline=0,
	             command=lambda :_qend(end))
    end_btn.grid(row=3, column=1, pady=(2,4))
    end_btn.configure(background='thistle3', fg="black")

    #=======================
    # find images button
    qfind_btn = Tkinter.Button(q_frame, text='Find Images', underline=0,
	             command=lambda :_qfind())
    qfind_btn.grid(row=4, column=0, pady=(2,4))
    qfind_btn.configure(background='thistle3', fg="black")

    #=======================
    # make movie
    self.movie2_btn = Tkinter.Button(q_frame, text='Make movie', underline=0,
	             command=lambda :_make_movie_withQdat())
    self.movie2_btn.grid(row=4, column=1, pady=(2,4))
    self.movie2_btn.configure(background='thistle3', fg="black")

    #=======================
    # process button
    qproc_btn = Tkinter.Button(q_frame, text='Process!', underline=0,
	             command=lambda :_proc(self))
    qproc_btn.grid(row=5, column=0, pady=(2,4))
    qproc_btn.configure(background='thistle3', fg="black")

    # ========================
    # close windows
    qdestroy_btn = Tkinter.Button(q_frame, text='Quit', underline=0,
	             command=lambda :_qquit(self))
    qdestroy_btn.grid(row=5, column=1, pady=(2,4))
    qdestroy_btn.configure(background='thistle3', fg="black")

    #=======================
    def _qstart(start):
        self.startdate = start.selection
        print("======================")
        print("Start Date:")
        print(self.startdate)
        print("======================")
        self.update()

    #=======================
    def _qend(end):
        print("======================")
        print("End Date:")
        self.enddate = end.selection
        print(self.enddate)
        print("======================")
        self.update()

    #=======================
    def _SetSitePick(master, v):
       sitelist = np.genfromtxt('sites.txt', dtype=str)

       self.sitepick= sitelist[v]
       print("site selected: "+self.sitepick)
       self.update()

    #=======================
    def _qquit(master):
        cv2.destroyAllWindows()
        master.destroy()

    #=======================
    def _SetTime(v):

       #self.sitepick= v
       if v==1:
          print("All times selected")
          self.idealtime = 'all'
       elif v==2:
          sitetimes = np.genfromtxt('ideal_time_per_site.txt', dtype=str, delimiter=',')
          self.idealtime = sitetimes[np.where(sitetimes[:,0] == self.sitepick)[0],1].tolist()[0].lstrip()
          print('Time selected: '+self.idealtime)
       elif v==3:
          print("Time selected: 06:00 -- 07:00")
          self.idealtime = '06:00'
       elif v==4:
          print("Time selected: 07:00 -- 08:00")
          self.idealtime = '07:00'
       elif v==5:
          print("Time selected: 08:00 -- 09:00")
          self.idealtime = '08:00'
       elif v==6:
          print("Time selected: 09:00 -- 10:00")
          self.idealtime = '09:00'
       elif v==7:
          print("Time selected: 10:00 -- 11:00")
          self.idealtime = '10:00'
       elif v==8:
          print("Time selected: 11:00 -- 12:00")
          self.idealtime = '11:00'
       elif v==9:
          print("Time selected: 12:00 -- 13:00")
          self.idealtime = '12:00'
       elif v==10:
          print("Time selected: 13:00 -- 14:00")
          self.idealtime = '13:00'
       elif v==11:
          print("Time selected: 14:00 -- 15:00")
          self.idealtime = '14:00'
       elif v==12:
          print("Time selected: 15:00 -- 16:00")
          self.idealtime = '15:00'
       elif v==13:
          print("Time selected: 16:00 -- 17:00")
          self.idealtime = '16:00'
       elif v==14:
          print("Time selected: 17:00 -- 18:00")
          self.idealtime = '17:00'
       elif v==15:
          print("Time selected: 18:00 -- 19:00")
          self.idealtime = '18:00'

       self.update()

    #=======================
    def _qfind():
        imagefolder = rootfolder + self.sitepick + os.sep

        # get a list of all the jpegs in the specified site folder
        types = ('*.jpg', '*.JPG', '*.jpeg')
        infiles = []
        for filetypes in types:
           infiles.extend(glob(imagefolder+filetypes))

        infiles = set(infiles)

        print("Number of files to search: "+str(len(infiles)))

        # determine the nearest gage and load the appropriate discharge data
        site = int(self.sitepick.split('RC')[-1].split('_')[0][:3])
        gages = np.asarray([0,30,61,87,166,225])
        nearest_gage = np.argmin(np.abs(site - gages))

        if self.datloaded == 0:
           print("Loading data from nearest gage ("+str(gages[nearest_gage])+" mile)")
           self.dat = load_gagedata(nearest_gage)
           self.datloaded = 1

        # get unix timestamps rfom the user selected start and end dates
        start_time = toTimestamp(self.startdate)+ 6 * 60 * 60
        end_time = toTimestamp(self.enddate)+ 6 * 60 * 60

        if self.idealtime != 'all':
           idealtime = int(self.idealtime[:2])

        # get unix timestamps and discharges of every file
        F=[]
        for filename in infiles:
            ext = os.path.splitext(filename)[1][1:]
            date = filename.split(os.sep)[-1].split('RC')[-1].split('_')[1]
            time = filename.split(os.sep)[-1].split('RC')[-1].split('_')[2].split('.'+ext)[0]
            timestamp = DT.datetime.strptime(date+' '+time, '%Y%m%d %H%M')
            if self.idealtime != 'all':
               if timestamp.hour == (idealtime) % 24 and timestamp.minute >= 30:
                  F.append(filename)
               elif timestamp.hour == (idealtime) % 24 and timestamp.minute <= 30:
                  F.append(filename)
            else:
               F.append(filename)

        # get unix timestamps and discharges of every file
        I = []; Q = []
        for filename in F: #infiles:
            ext = os.path.splitext(filename)[1][1:]
            date = filename.split(os.sep)[-1].split('RC')[-1].split('_')[1]
            time = filename.split(os.sep)[-1].split('RC')[-1].split('_')[2].split('.'+ext)[0]
            image_time = toTimestamp(DT.datetime.strptime(date+' '+time, '%Y%m%d %H%M'))+ 6 * 60 * 60
            I.append(image_time)
            # add 6 hours (mst to gmt)
            Q.append(np.interp(image_time,self.dat['timeunix'],self.dat['Qcfs']))

        Q = np.asarray(Q)
        I = np.asarray(I)

        indices = np.where((I>=start_time) & (I<=end_time) & (Q>=self.Nvar.get()-100) & (Q<=self.Nvar.get()+100))[0]
        print("Number of files within time window and near specified discharge: "+str(len(indices)))

        self.imagefiles = np.asarray(F)[indices] #infiles
        self.update()

    #==============================================================
    #==============================================================
    #========END 2nd tab

    #==============================================================
    #==============================================================
    #========START 3rd tab

    # Populate the second pane. Note that the content doesn't really matter
    t_frame = Tkinter.Frame(nb)
    nb.add(t_frame, text='Process Images Based on Site/Time')#, state='disabled')

    t_frame.configure(background='purple')

    MSG1_btn = Tkinter.Button(t_frame, text = "Instructions", command = hello1_alt)
    MSG1_btn.grid(row=0, column=0, pady=(2,4))
    MSG1_btn.configure(background='thistle3', fg="black")

    t_frame.rowconfigure(1, weight=1)
    t_frame.columnconfigure((0,1), weight=1, uniform=1)

    #=======================
    #menu for site
    self.bb=  Tkinter.Menubutton ( t_frame, text="Choose Site", relief=Tkinter.RAISED)
    self.bb.grid(column = 0, row = 1, pady=(2,4))
    self.bb.menu  =  Tkinter.Menu ( self.bb, tearoff = 1 , background='PaleVioletRed1', fg="black")

    sitelist = np.genfromtxt('sites.txt', dtype=str)

    submenu1 = Menu(self.bb.menu)
    for site in xrange(11):
       submenu1.add_command(label=sitelist[site], command = lambda v=site: _SetSitePick(master, v),  font=('Arial', 10, 'bold', 'italic'))

    self.bb.menu.add_cascade(label='Upper Marble Canyon', menu=submenu1, underline=0)

    submenu2 = Menu(self.bb.menu)
    for site in xrange(12,34):
       submenu2.add_command(label=sitelist[site], command = lambda v=site: _SetSitePick(master, v),  font=('Arial', 10, 'bold', 'italic'))

    self.bb.menu.add_cascade(label='Lower Marble Canyon', menu=submenu2, underline=0)

    submenu3 = Menu(self.bb.menu)
    for site in xrange(35, 47):
       submenu3.add_command(label=sitelist[site], command = lambda v=site: _SetSitePick(master, v),  font=('Arial', 10, 'bold', 'italic'))

    self.bb.menu.add_cascade(label='Eastern Grand Canyon', menu=submenu3, underline=0)

    submenu4 = Menu(self.bb.menu)
    for site in xrange(48,71):
       submenu4.add_command(label=sitelist[site], command = lambda v=site: _SetSitePick(master, v),  font=('Arial', 10, 'bold', 'italic'))

    self.bb.menu.add_cascade(label='Western Grand Canyon', menu=submenu4, underline=0)

    self.bb["menu"]  =  self.bb.menu
    self.bb.configure(background='thistle3', fg="black")

    #=======================
    #menu for time
    self.Mb=  Tkinter.Menubutton ( t_frame, text="Set time", relief=Tkinter.RAISED)
    self.Mb.grid(column = 1, row = 1, pady=(2,4))
    self.Mb.menu  =  Tkinter.Menu ( self.Mb, tearoff = 1 , background='orchid2', fg="black" )
    self.Mb.menu.add_command(label="All times", command = lambda v=1: _SetTime(v))
    self.Mb.menu.add_command(label="Ideal time for site", command = lambda v=2: _SetTime(v))
    self.Mb.menu.add_command(label="06:00 -- 07:00", command = lambda v=3: _SetTime(v))
    self.Mb.menu.add_command(label="07:00 -- 08:00", command = lambda v=4: _SetTime(v))
    self.Mb.menu.add_command(label="08:00 -- 09:00", command = lambda v=5: _SetTime(v))
    self.Mb.menu.add_command(label="09:00 -- 10:00", command = lambda v=6: _SetTime(v))
    self.Mb.menu.add_command(label="10:00 -- 11:00", command = lambda v=7: _SetTime(v))
    self.Mb.menu.add_command(label="11:00 -- 12:00", command = lambda v=8: _SetTime(v))
    self.Mb.menu.add_command(label="12:00 -- 13:00", command = lambda v=9: _SetTime(v))
    self.Mb.menu.add_command(label="13:00 -- 14:00", command = lambda v=10: _SetTime(v))
    self.Mb.menu.add_command(label="14:00 -- 15:00", command = lambda v=11: _SetTime(v))
    self.Mb.menu.add_command(label="15:00 -- 16:00", command = lambda v=12: _SetTime(v))
    self.Mb.menu.add_command(label="16:00 -- 17:00", command = lambda v=13: _SetTime(v))
    self.Mb.menu.add_command(label="17:00 -- 18:00", command = lambda v=14: _SetTime(v))
    self.Mb.menu.add_command(label="18:00 -- 19:00", command = lambda v=15: _SetTime(v))
    self.Mb["menu"]  =  self.Mb.menu
    self.Mb.configure(background='thistle3', fg="black")

    #=======================
    # start date button
    tstart = Calendar2(t_frame)
    tstart.grid(row=2, column=0, pady=(2,4))

    self.startdate = tstart.selection

    tstart_btn = Tkinter.Button(t_frame, text='Set Start Date', underline=0,
	             command=lambda :_tstart(tstart))
    tstart_btn.grid(row=3, column=0, pady=(2,4))
    tstart_btn.configure(background='thistle3', fg="black")

    #=======================
    # end date button
    tend = Calendar2(t_frame)
    tend.grid(row=2, column=1, pady=(2,4))

    self.enddate = tend.selection

    tend_btn = Tkinter.Button(t_frame, text='Set End Date', underline=0,
	             command=lambda :_tend(tend))
    tend_btn.grid(row=3, column=1, pady=(2,4))
    tend_btn.configure(background='thistle3', fg="black")

    #=======================
    # find images button
    tfind_btn = Tkinter.Button(t_frame, text='Find Images', underline=0,
	             command=lambda :_tfind())
    tfind_btn.grid(row=4, column=0, pady=(2,4))
    tfind_btn.configure(background='thistle3', fg="black")

    #=======================
    # make movie
    self.movie3_btn = Tkinter.Button(t_frame, text='Make movie', underline=0,
	             command=lambda :_make_movie())
    self.movie3_btn.grid(row=4, column=1, pady=(2,4))
    self.movie3_btn.configure(background='thistle3', fg="black")

    #=======================
    # process button
    tproc_btn = Tkinter.Button(t_frame, text='Process!', underline=0,
	             command=lambda :_proc(self))
    tproc_btn.grid(row=5, column=0, pady=(2,4))
    tproc_btn.configure(background='thistle3', fg="black")

    # ========================
    # close windows
    tdestroy_btn = Tkinter.Button(t_frame, text='Quit', underline=0,
	             command=lambda :_qquit(self))
    tdestroy_btn.grid(row=5, column=1, pady=(2,4))
    tdestroy_btn.configure(background='thistle3', fg="black")

    #=======================
    def _tstart(tstart):
        self.startdate = tstart.selection
        print("======================")
        print("Start Date:")
        print(self.startdate)
        print("======================")
        self.update()

    #=======================
    def _tend(tend):
        print("======================")
        print("End Date:")
        self.enddate = tend.selection
        print(self.enddate)
        print("======================")
        self.update()


    #=======================
    def _tfind():
        imagefolder = rootfolder + self.sitepick + os.sep

        # get a list of all the jpegs in the specified site folder
        types = ('*.jpg', '*.JPG', '*.jpeg')
        infiles = []
        for filetypes in types:
           infiles.extend(glob(imagefolder+filetypes))

        infiles = set(infiles)

        print("Number of files to search: "+str(len(infiles)))

        # get unix timestamps rfom the user selected start and end dates
        start_time = toTimestamp(self.startdate)+ 6 * 60 * 60
        end_time = toTimestamp(self.enddate)+ 6 * 60 * 60

        if self.idealtime != 'all':
           idealtime = int(self.idealtime[:2])

        # get unix timestamps and discharges of every file
        F=[]
        for filename in infiles:
            ext = os.path.splitext(filename)[1][1:]
            date = filename.split(os.sep)[-1].split('RC')[-1].split('_')[1]
            time = filename.split(os.sep)[-1].split('RC')[-1].split('_')[2].split('.'+ext)[0]
            timestamp = DT.datetime.strptime(date+' '+time, '%Y%m%d %H%M')
            if self.idealtime != 'all':
               if timestamp.hour == (idealtime) % 24 and timestamp.minute >= 30:
                  F.append(filename)
               elif timestamp.hour == (idealtime) % 24 and timestamp.minute <= 30:
                  F.append(filename)
            else:
               F.append(filename)

        # get unix timestamps and discharges of every file
        I = []
        for filename in F:
            ext = os.path.splitext(filename)[1][1:]
            date = filename.split(os.sep)[-1].split('RC')[-1].split('_')[1]
            time = filename.split(os.sep)[-1].split('RC')[-1].split('_')[2].split('.'+ext)[0]
            timestamp = DT.datetime.strptime(date+' '+time, '%Y%m%d %H%M')
            #if self.idealtime != 'all':
            #   if timestamp.hour == (idealtime) % 24 and timestamp.minute >= 30:
            #      image_time = toTimestamp(timestamp)+ 6 * 60 * 60
            #      I.append(image_time) # add 6 hours (mst to gmt)
            #   elif timestamp.hour == (idealtime) % 24 and timestamp.minute <= 30:
            #      image_time = toTimestamp(timestamp)+ 6 * 60 * 60
            #      I.append(image_time) # add 6 hours (mst to gmt)
            #else:
            image_time = toTimestamp(timestamp)+ 6 * 60 * 60
            I.append(image_time) # add 6 hours (mst to gmt)

        I = np.asarray(I)

        indices = np.where((I>=start_time) & (I<=end_time))[0]
        print("Number of files within time window: "+str(len(indices)))

        self.imagefiles = np.asarray(F)[indices]
        self.update()

    #==============================================================
    #==============================================================
    #========END 3rd tab

    # start app
    master.mainloop()

# =========================================================
# =========================================================
if __name__ == '__main__':

   gui()


#    #=======================
#    # discharge
#    self.Nvar = Tkinter.DoubleVar()
#    Nscale = Tkinter.Scale( t_frame, variable = self.Nvar, from_=5000, to=45000, resolution=1000, tickinterval=1000, label = 'Discharge' )
#    Nscale.set(8000)
#    Nscale.grid(row=1, column=1,  pady=(2,4))
#    Nscale.configure(background='thistle3', fg="black")

#    #=======================
#    def _qget_images(): #(master, v):
#        self.qimagefiles = askopenfilename(filetypes = [ ("Image Files", ("*.jpg", "*.JPG", '*.jpeg')), ("TIF",('*.tif', '*.tiff'))] , multiple=True)

#        for k in xrange(len(self.qimagefiles)):
#           print(self.qimagefiles[k])
#        self.qfolder = os.path.dirname(self.qimagefiles[0])

#        #self.q_son_btn.configure(fg='thistle3', background="black")
#
#        self.update()

#	#=======================
#	def _qstart(start):
#           self.startdate = start.selection
#           print("======================")
#           print("Start Date:")
#           print(self.startdate)
#           print("======================")
#           self.update()

#	#=======================
#	def _qend(end):
#           print("======================")
#           print("End Date:")
#           self.enddate = end.selection
#           print(self.enddate)
#           print("======================")
#           self.update()

#	#=======================
#	def _SetSitePick(master, v):
#           sitelist = np.genfromtxt('sites.txt', dtype=str)
#
#	   self.sitepick= sitelist[v]
#           print("site selected: "+self.sitepick)

#	   #self.bb.configure(fg='thistle3', background="black")
#	   self.update()

#	#=======================
#	def _qquit(master):
#           cv2.destroyAllWindows()
#           master.destroy()

#	#=======================
#	def _qfind():
#           imagefolder = rootfolder + self.sitepick + os.sep

#           # get a list of all the jpegs in the specified site folder
#           types = ('*.jpg', '*.JPG', '*.jpeg')
#           infiles = []
#           for filetypes in types:
#              infiles.extend(glob(imagefolder+filetypes))

#           print("Number of files to search: "+str(len(infiles)))

#           # determine the nearest gage and load the appropriate discharge data
#           site = int(self.sitepick.split('RC')[-1].split('_')[0][:3])
#           gages = np.asarray([0,30,61,87,166,225])
#           nearest_gage = np.argmin(np.abs(site - gages))

#           print("Loading data from nearest gage ("+str(gages[nearest_gage])+" mile)")
#           dat = load_gagedata(nearest_gage)

#           # get unix timestamps rfom the user selected start and end dates
#           #start_time = toTimestamp(DT.datetime.strptime(self.startdate, '%Y-%m-%d %H:%M:%S'))+ 6 * 60 * 60
#           #end_time = toTimestamp(DT.datetime.strptime(self.enddate, '%Y-%m-%d %H:%M:%S'))+ 6 * 60 * 60
#           start_time = toTimestamp(self.startdate)+ 6 * 60 * 60
#           end_time = toTimestamp(self.enddate)+ 6 * 60 * 60
#
#           # get unix timestamps and discharges of every file
#           I = []; Q = []
#           for filename in infiles:
#               ext = os.path.splitext(filename)[1][1:]
#               date = filename.split(os.sep)[-1].split('RC')[-1].split('_')[1]
#               time = filename.split(os.sep)[-1].split('RC')[-1].split('_')[2].split('.'+ext)[0]
#               image_time = toTimestamp(DT.datetime.strptime(date+' '+time, '%Y%m%d %H%M'))+ 6 * 60 * 60
#               I.append(image_time)
#               # add 6 hours (mst to gmt)
#               Q.append(np.interp(image_time,dat['timeunix'],dat['Qcfs']))

#           Q = np.asarray(Q)
#           I = np.asarray(I)

#           indices = np.where((I>=start_time) & (I<=end_time) & (Q>=self.Nvar.get()-100) & (Q<=self.Nvar.get()+100))[0]
#           print("Number of files within time window and near specified discharge: "+str(len(indices)))

#           self.qimagefiles = np.asarray(infiles)[indices]
#	   self.update()

#	#=======================
#	def _qget_images(): #(master, v):
#	    self.qimagefiles = askopenfilename(filetypes = [ ("Image Files", ("*.jpg", "*.JPG", '*.jpeg')), ("TIF",('*.tif', '*.tiff'))] , multiple=True)

#	    for k in xrange(len(self.qimagefiles)):
#	       print(self.qimagefiles[k])
#	    self.qfolder = os.path.dirname(self.qimagefiles[0])
#
#	    #self.son_btn.configure(fg='thistle3', background="black")

#	    self.q_son_btn.configure(fg='thistle3', background="black")
#
#	    self.update()

               #img = imresize(cv2.imread(filename),scale) #resize image so quarter size
               #imagehsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
               #im = imresize(cv2.imread(filename,0),scale) #resize image so quarter size
               #la = cv2.Laplacian(im,cv2.CV_64F)
               # get std and mean through stndard deviation, fast thru convolution
               #m1, s1 = std_convoluted(im, .5)
               #m2, s2 = std_convoluted(im, .25)
