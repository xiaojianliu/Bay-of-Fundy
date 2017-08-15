# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 14:20:55 2017

@author: hxu
"""
import sys
import datetime as dt
from matplotlib.path import Path
import netCDF4
from dateutil.parser import parse
import numpy as np
import math
import pandas as pd
from datetime import datetime, timedelta
from math import radians, cos, sin, atan, sqrt  
from matplotlib.dates import date2num,num2date
import matplotlib.pyplot as plt
FN='necscoast_worldvec.dat'
CL=np.genfromtxt(FN,names=['lon','lat'])
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
pgon=plt.Polygon([[-65.82,45.037990],[-65.75,44.9247],[-65.7497,44.799],[-65.906,44.6582],[-66.1024,44.5559],[-66.2821,44.5023],[-66.47165,44.51],[-66.628,44.581],[-66.6838,44.7116],[-66.6838,44.8044],[-66.5956,44.9247],[-66.4290,45.03568],[-66.2657,45.088745],[-66.11,45.109],[-65.9620,45.0749]],color='g',alpha=0.5)
ax.add_patch(pgon)
plt.plot(CL['lon'],CL['lat'])
plt.axis([-68,-65,44,45.33])
lon=np.linspace(-67,-65.5,20)
lat=np.linspace(44.4,45.3,20)
lo=[]
la=[]
for a in np.arange(len(lon)):
    for b in np.arange(len(lat)):
        lo.append(lon[a])
        la.append(lat[b])

lo1=[]
la1=[]
for a in np.arange(3,len(lat)-5,1):
    lo1.append(lon[10])
    la1.append(lat[a])
for a in np.arange(3,len(lat)-6,1):
    lo1.append(lon[9])
    la1.append(lat[a])
for a in np.arange(4,len(lat)-5,1):
    lo1.append(lon[11])
    la1.append(lat[a])
for a in np.arange(5,len(lat)-5,1):
    lo1.append(lon[12])
    la1.append(lat[a])
for a in np.arange(6,len(lat)-5,1):
    lo1.append(lon[13])
    la1.append(lat[a])
for a in np.arange(7,len(lat)-6,1):
    lo1.append(lon[14])
    la1.append(lat[a])
for a in np.arange(8,len(lat)-7,1):
    lo1.append(lon[15])
    la1.append(lat[a])
for a in np.arange(3,len(lat)-6,1):
    lo1.append(lon[8])
    la1.append(lat[a])
for a in np.arange(3,len(lat)-7,1):
    lo1.append(lon[7])
    la1.append(lat[a])
for a in np.arange(4,len(lat)-8,1):
    lo1.append(lon[6])
    la1.append(lat[a])
for a in np.arange(5,len(lat)-9,1):
    lo1.append(lon[5])
    la1.append(lat[a])
plt.scatter(lo1,la1)  
plt.savefig('xx',dpi=300)
from matplotlib.path import Path
import matplotlib.patches as patches

vertices = [(-65.82,45.037990),(-65.75,44.9247),(-65.7497,44.799),(-65.906,44.6582),(-66.1024,44.5559),(-66.2821,44.5023),(-66.47165,44.51),(-66.628,44.581),(-66.6838,44.7116),(-66.6838,44.8044),(-66.5956,44.9247),(-66.4290,45.03568),(-66.2657,45.088745),(-66.11,45.109),(-65.9620,45.0749)]
'''
codes = [Path.MOVETO, Path.LINETO,
         Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
'''
path = Path(vertices)#, codes)
for a in np.arange(len(lo1)):
    print path.contains_points([[lo1[a], la1[a]]])


#print path.contains_points([[3, 3], [0, 0]])
