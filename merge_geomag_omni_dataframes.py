
# coding: utf-8

# colnames

# In[ ]:




# In[1]:

from obspy.core import UTCDateTime

import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd

#
#
#  READING THE GEOMAGNETIC FIELD DATA (pkl FILE) X, Y, Z and F components
#
#

#dir = '/data/st/geomag_2015_2016_xyzf/'
dir = './'
filenames = ['X_2016_minutes.pkl', 'Y_2016_minutes.pkl','Z_2016_minutes.pkl','F_2016_minutes.pkl']
ncomps = len(filenames)

# First get number of rows and number of observatories
with open(dir+'X_2016_minutes.pkl', 'rb') as f2:
#        compseries = pickle.load(f2,encoding='latin1')
        compseries = pickle.load(f2)
comp=compseries[0]
times = comp.times()
numrows = len(times)
n_observatories = len(compseries)
colnames = ["" for x in range(ncomps*n_observatories)]

# Make UTC timestamps
UTCtimes = np.zeros(numrows, dtype=float)
for c in range(numrows):
    UTCtimes[c] = (UTCDateTime(comp.__dict__['stats']['starttime']) + times[c]).timestamp

# Now loop through pickle files and put all into a 2D np array
geo = np.zeros([numrows,ncomps*len(compseries)])
# Geomag array   
for c in range(len(filenames)):
    with open(dir+filenames[c], 'rb') as f2:
        #compseries = pickle.load(f2,encoding='latin1')
        compseries = pickle.load(f2)
    for o in range(n_observatories):
        geo[:,ncomps*o + c] = compseries[o].data
        colnames[ncomps*o + c] = compseries[o].__dict__['stats']['station']+'_'+compseries[o].__dict__['stats']['channel']

# Put into pandas DataFrame
geomagdf = pd.DataFrame(geo, columns=colnames)
geomagdf.insert(0, 'Date', pd.Series(np.array(UTCtimes)))


# In[2]:

geomagdf


# In[3]:

##################################################################
###                  READING OMNI DATA 
##################################################################
omnirows = ['Year', 'Day', 'Hour', 'Minute', 'Field magnitude average nT', 'BX nT (GSE, GSM)', 
            'BY, nT (GSE)', 'BZ, nT (GSE)', 'BY, nT (GSM)', 'BZ, nT (GSM)', 'RMS SD B scalar, nT', 
            'RMS SD field vector, nT', 'Speed, km/s', 'Vx Velocity,km/s', 'Vy Velocity, km/s', 
            'Vz Velocity, km/s', 'Proton Density, n/cc', 'Temperature, K', 'Flow pressure, nPa', 
            'Electric field, mV/m', 'Total Plasma beta', 'Alfven mach number', 'Magnetosonic Mach number', 
            'S/C Xgse Re', 'S/C Ygse Re', 'S/c Zgse Re', 'BSN location Xgse Re',
            'BSN location Ygse Re','BSN location Zgse Re','AE-index, nT','AL-index, nT',
            'AU-index, nT', 'PCN-index']


# In[4]:

#dff = pd.DataFrame(omniarr[0:len(omnirows),:], index=[omnirows])
#omnigeo= pd.concat([df, dff])
#omnigeo.shape


# In[5]:

colnames=["Year", "Day", "Hour", "Minute","ID IMF Spacecraft", "IF SW Plasma Spacecraft",
          "#points in IMF avg", "#points in plasma avgs", "Percent interp", "Timeshift (sec)", 
          "RMS, timeshift", "RMS, phase front normal", "Time btwn obs (sec)", 
          "Field mag avg, nT", "Bx, nT (GSE, GSM)", "By, nT (GSE,GSM)", "Bz, nT (GSE)", "By, nT (GSM)", "Bz, nT (GSM)",
          "RMS SD B scalar, nT", "RMS SD field vector, nT", "Flow speed, km/s", "Vx, km/s, GSE", "Vy, km/s, GSE", "Vz, km/s, GSE", 
          "Proton density, n/cc", "Temperture, K", "Flow pressure, nPa", "Electric Field, mV/m", "Plasma beta", 
          "Alfven mach number", "X(s/c), GSE, Re", "Y(s/c), GSE, Re", "Z(s/c), GSE, Re",
          "BSN location, Xgse, Re", "BSN location, Ygse, Re", "BSN location, Zgse, Re", 
          "AE-index, nT", "AL-index, nT", "AU-index, nT", "SYM/D index, nT", "SYM/H index, nT", 
          "ASY/D index, nT", "ASY/H index, nT", "PC(N) index, nT", 
          "Magnetosonic mach number"]
print(len(colnames))
#omnidir = '/data/st/omni/high_res_omni/'
omnidir = './'
omnidf=pd.read_csv(omnidir+'omni_min2016.asc',delimiter='\s+',skiprows=0,names=colnames)

#          I4      1 ... 365 or 366
#01#Year
#02#Day
#03#Hour                            I3      0 ... 23
#04#Minute                          I3      0 ... 59 at start of average
#05#ID for IMF spacecraft           I3      See  footnote D below
#06#ID for SW Plasma spacecraft     I3      See  footnote D below
#07## of points in IMF averages     I4
#08## of points in Plasma averages  I4
#09#Percent interp                  I4      See  footnote A below
#10#Timeshift, sec                  I7
#11#RMS, Timeshift                  I7
#12#RMS, Phase front normal         F6.2    See Footnotes E, F below
#13#Time btwn observations, sec     I7      DBOT1, See Footnote C below
#14#Field magnitude average, nT     F8.2
#15#Bx, nT (GSE, GSM)               F8.2
#16#By, nT (GSE)                    F8.2
#17#Bz, nT (GSE)                    F8.2
#18#By, nT (GSM)                    F8.2    Determined from post-shift GSE components
#19#Bz, nT (GSM)                    F8.2    Determined from post-shift GSE components
#20#RMS SD B scalar, nT             F8.2
#21#RMS SD field vector, nT         F8.2    See  footnote E below
#22#Flow speed, km/s                F8.1
#23#Vx Velocity, km/s, GSE          F8.1
#24#Vy Velocity, km/s, GSE          F8.1
#25#Vz Velocity, km/s, GSE          F8.1
#26#Proton Density, n/cc            F7.2
#27#Temperature, K                  F9.0
#28#Flow pressure, nPa              F6.2    See  footnote G below
#29#Electric field, mV/m            F7.2    See  footnote G below
#30#Plasma beta                     F7.2    See  footnote G below
#31#Alfven mach number              F6.1    See  footnote G below
#32#X(s/c), GSE, Re                 F8.2
#33#Y(s/c), GSE, Re                 F8.2
#34#Z(s/c), GSE, Re                 F8.2
#35#BSN location, Xgse, Re          F8.2    BSN = bow shock nose
#36#BSN location, Ygse, Re          F8.2
#37#BSN location, Zgse, Re          F8.2
#38#AE-index, nT                    I6      See  footnote H below
#39#AL-index, nT                    I6      See  footnote H below
#40#AU-index, nT                    I6      See  footnote H below
#41#SYM/D index, nT                 I6      See  footnote H below
#42#SYM/H index, nT                 I6      See  footnote H below
#43#ASY/D index, nT                 I6      See  footnote H below
#44#ASY/H index, nT                 I6      See  footnote H below
#45#PC(N) index,                    F7.2    See  footnote I below
#46#Magnetosonic mach number        F5.1    See  Footnote K below
#(2I4,4I3,3I4,2I7,F6.2,I7, 8F8.2,4F8.1,F7.2,F9.0,F6.2,2F7.2,F6.1,6F8.2,7I6,F7.2, F5.1)


# In[6]:

# Create DataFrame for UTCDateTime entries
times = []
for c in range(len(omnidf['Year'])):
    times.append((UTCDateTime(
        "{0}-{1:03d}T{2:02d}:{3:02d}:00.0".format(omnidf['Year'][c],
                                                  omnidf['Day'][c],
                                                  omnidf['Hour'][c],
                                                  omnidf['Minute'][c]))).timestamp)

date_df = pd.DataFrame(np.array(times), columns=['Date'])
# Insert column into omnidf
omnidf.insert(0, 'Date', pd.Series(np.array(times)))


# In[7]:

omnidf


# In[8]:

print(omnidf.shape)
print(geomagdf.shape)


# In[9]:

df = omnidf.merge(geomagdf, left_on='Date', right_on='Date', how='inner')


# In[10]:

df


# In[11]:

df.columns


# In[12]:

times    = df.loc[:,'Date']
raw_data = df.loc[:,['Field mag avg, nT', 'Bx, nT (GSE, GSM)', 'By, nT (GSE,GSM)',
       'Bz, nT (GSE)', 'By, nT (GSM)', 'Bz, nT (GSM)', 'RMS SD B scalar, nT',
       'RMS SD field vector, nT', 'Flow speed, km/s', 'Vx, km/s, GSE',
       'Vy, km/s, GSE', 'Vz, km/s, GSE', 'Proton density, n/cc',
       'Temperture, K', 'Flow pressure, nPa', 'Electric Field, mV/m',
       'Plasma beta', 'Alfven mach number', 'BOU_X', 'BOU_Y', 'BOU_Z', 'BOU_F', 'BRW_X', 'BRW_Y',
       'BRW_Z', 'BRW_F', 'BSL_X', 'BSL_Y', 'BSL_Z', 'BSL_F', 'CMO_X',
       'CMO_Y', 'CMO_Z', 'CMO_F', 'DED_X', 'DED_Y', 'DED_Z', 'DED_F',
       'FRD_X', 'FRD_Y', 'FRD_Z', 'FRD_F', 'FRN_X', 'FRN_Y', 'FRN_Z',
       'FRN_F', 'GUA_X', 'GUA_Y', 'GUA_Z', 'GUA_F', 'HON_X', 'HON_Y',
       'HON_Z', 'HON_F', 'NEW_X', 'NEW_Y', 'NEW_Z', 'NEW_F', 'SHU_X',
       'SHU_Y', 'SHU_Z', 'SHU_F', 'SIT_X', 'SIT_Y', 'SIT_Z', 'SIT_F',
       'SJG_X', 'SJG_Y', 'SJG_Z', 'SJG_F', 'TUC_X', 'TUC_Y', 'TUC_Z',
       'TUC_F']]


# In[13]:

print(raw_data.values.shape)
print(times.values.shape)


# In[ ]:



