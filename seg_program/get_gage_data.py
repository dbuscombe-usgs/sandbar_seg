
import numpy as np
from glob import glob
import datetime as DT
import cPickle as pickle


import calendar

def toTimestamp(d):
  return calendar.timegm(d.timetuple())


def timeConv(x):
    try:
        return DT.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f') + DT.timedelta(0,6*60*60) #add 6 hours
    except ValueError as err:
        return DT.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') + DT.timedelta(0,6*60*60) #6 hours

# 0-mile
time = np.genfromtxt(glob('rm0*.tsv')[0], dtype='object', delimiter='\t', skip_header=1, usecols=(0), converters = {0: timeConv})

Qcfs= np.genfromtxt(glob('rm0*.tsv')[0], skip_header=1, usecols=(1), dtype='float', delimiter='\t') 

timeunix = np.array([toTimestamp(d) for d in time]) 

pickle.dump( {"time": time, "timeunix": timeunix, "Qcfs": Qcfs}, open( "rm0_time_Qcfs.p", "wb" ), protocol=pickle.HIGHEST_PROTOCOL )


## 30 mile
#time = np.genfromtxt(glob('rm30*.tsv')[0], dtype='object', delimiter='\t', skip_header=1, usecols=(0), converters = {0: timeConv})

#Qcfs= np.genfromtxt(glob('rm30*.tsv')[0], skip_header=1, usecols=1, dtype='float', delimiter='\t') 

#timeunix = np.array([toTimestamp(d) for d in time]) 

#pickle.dump( {"time": time, "timeunix": timeunix, "Qcfs": Qcfs}, open( "rm30_time_Qcfs.p", "wb" ), protocol=pickle.HIGHEST_PROTOCOL )


## 61 mile
##time = np.genfromtxt(glob('rm61*.tsv')[0], dtype='object', delimiter='\t', skip_header=1, usecols=(0), converters = {0: timeConv})

##Qcfs= np.genfromtxt(glob('rm61*.tsv')[0], skip_header=1, usecols=1, dtype='float', delimiter='\t') 

##timeunix = np.array([toTimestamp(d) for d in time]) 

##pickle.dump( {"time": time, "timeunix": timeunix, "Qcfs": Qcfs}, open( "rm61_time_Qcfs.p", "wb" ), protocol=pickle.HIGHEST_PROTOCOL )

## 87 mile
##time = np.genfromtxt(glob('rm87*.tsv')[0], dtype='object', delimiter='\t', skip_header=1, usecols=(0), converters = {0: timeConv})

##Qcfs= np.genfromtxt(glob('rm87*.tsv')[0], skip_header=1, usecols=1, dtype='float', delimiter='\t') 

##timeunix = np.array([toTimestamp(d) for d in time]) 

##pickle.dump( {"time": time, "timeunix": timeunix, "Qcfs": Qcfs}, open( "rm87_time_Qcfs.p", "wb" ), protocol=pickle.HIGHEST_PROTOCOL )


## 166 mile
##time = np.genfromtxt(glob('rm166*.tsv')[0], dtype='object', delimiter='\t', skip_header=1, usecols=(0), converters = {0: timeConv})

##Qcfs= np.genfromtxt(glob('rm166*.tsv')[0], skip_header=1, usecols=1, dtype='float', delimiter='\t') 

##timeunix = np.array([toTimestamp(d) for d in time]) 

##pickle.dump( {"time": time, "timeunix": timeunix, "Qcfs": Qcfs}, open( "rm166_time_Qcfs.p", "wb" ), protocol=pickle.HIGHEST_PROTOCOL )


## 225 mile
##time = np.genfromtxt(glob('rm225*.tsv')[0], dtype='object', delimiter='\t', skip_header=1, usecols=(0), converters = {0: timeConv})

##Qcfs= np.genfromtxt(glob('rm225*.tsv')[0], skip_header=1, usecols=1, dtype='float', delimiter='\t') 

##timeunix = np.array([toTimestamp(d) for d in time]) 

##pickle.dump( {"time": time, "timeunix": timeunix, "Qcfs": Qcfs}, open( "rm225_time_Qcfs.p", "wb" ), protocol=pickle.HIGHEST_PROTOCOL )


#def generate_url(name_of_station):

#    url1 = 'http://waterservices.usgs.gov/nwis/iv/?'
#    url2 = 'format=rdb'
#    url3 = 'sites=' + name_of_station
#    url4 = 'startDT=2015-01-01'
#    url5 = 'endDT=2016-01-01'
#    url6 = 'parameterCd=00060,00065'

#    url = url1 + url2 + '&' + url3 + '&' + url4 + '&' + url5 + '&' + url6
#    
#    return url


#lees_ferry = '09380000'
#grand_canyon = '09402500'
#diamond_creek = '09404200'

#generate_url(lees_ferry)
