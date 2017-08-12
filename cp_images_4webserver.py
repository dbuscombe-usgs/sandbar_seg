
import cPickle as pickle
import datetime as DT
import os
from glob import glob
import calendar
import numpy as np

rootfolder = '/run/media/dbuscombe/MASTER/GCMRC/SANDBAR_REMOTECAMERAS/' #root folder where images are located

Nvar = 11000 # ideal discharge

sitepick = 'RC0307Rf'

idealtime = '14:00'
idealtime = int(idealtime[:2])

startdate = -1 #-1 == start of record
enddate = 99 # 99 == end of record

##============================================
def toTimestamp(d):
  return calendar.timegm(d.timetuple())


##======================================================
def load_gagedata(nearest_gage):

   if os.name=='nt':
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

   else:
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


#=======================
def qfind():
    imagefolder = rootfolder + sitepick + os.sep

    # get a list of all the jpegs in the specified site folder
    types = ('*.jpg', '*.JPG', '*.jpeg')
    infiles = []
    for filetypes in types:
       infiles.extend(glob(imagefolder+filetypes))

    infiles = sorted(infiles)

    print("Number of files to search: "+str(len(infiles)))

    # determine the nearest gage and load the appropriate discharge data
    site = int(sitepick.split('RC')[-1].split('_')[0][:3])
    gages = np.asarray([0,30,61,87,166,225])
    nearest_gage = np.argmin(np.abs(site - gages))

    print("Loading data from nearest gage ("+str(gages[nearest_gage])+" mile)")
    dat = load_gagedata(nearest_gage)

    #apply shift in hours equivalent to 4 miles per hour to nearest upstream gage
    4*site-gages[nearest_gage]


    # get unix timestamps rfom the user selected start and end dates

    if startdate==-1:

        ext = os.path.splitext(infiles[0])[1][1:]
        date = infiles[0].split(os.sep)[-1].split('RC')[-1].split('_')[1]
        time = infiles[0].split(os.sep)[-1].split('RC')[-1].split('_')[2].split('.'+ext)[0]
        timestamp = DT.datetime.strptime(date+' '+time, '%Y%m%d %H%M')

        start_time = toTimestamp(timestamp)+ 6 * 60 * 60

    else:
        start_time = toTimestamp(startdate)+ 6 * 60 * 60


    if enddate == 99:
       
        ext = os.path.splitext(infiles[-1])[1][1:]
        date = infiles[-1].split(os.sep)[-1].split('RC')[-1].split('_')[1]
        time = infiles[-1].split(os.sep)[-1].split('RC')[-1].split('_')[2].split('.'+ext)[0]
        timestamp = DT.datetime.strptime(date+' '+time, '%Y%m%d %H%M')

        end_time = toTimestamp(timestamp)+ 6 * 60 * 60

    else:
        end_time = toTimestamp(enddate)+ 6 * 60 * 60

    # get unix timestamps and discharges of every file
    F=[]
    for filename in infiles:
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
        Q.append(np.interp(image_time,dat['timeunix'],dat['Qcfs']))
    Q = np.asarray(Q)
    I = np.asarray(I)

    indices = np.where((I>=start_time) & (I<=end_time) & (Q>=Nvar-200) & (Q<=Nvar+200))[0]
    print("Number of files within time window and near specified discharge: "+str(len(indices)))
    imagefiles = np.asarray(F)[indices] #infiles



