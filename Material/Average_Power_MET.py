# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 18:50:55 2019

@author: Peter Husemeyer
"""

#%% Import functions

import numpy as np

#%% INPUT DATA
"""
Brad, enter the distance that the player did in the 4 speed bands here in this order

# 0 m/s < distance (m) < 3 m/s
# 3 m/s < distance (m) < 5 m/s
# 5 m/s < distance (m) < 7 m/s
# 7 m/s < distance (m) < 10 m/s
"""

SPEED_BANDS_1  = np.array([555, 555, 555, 555]) #Walker
SPEED_BANDS_2  = np.array([2255, 1008, 457, 378]) #Williams
SPEED_BANDS_3  = np.array([555, 555, 555, 555]) #Battye
SPEED_BANDS_4  = np.array([0, 0, 0, 0])
SPEED_BANDS_5  = np.array([0, 0, 0, 0])
SPEED_BANDS_6  = np.array([0, 0, 0, 0])
SPEED_BANDS_7  = np.array([0, 0, 0, 0])

""" PLAYER WEIGHT """
WEIGHT_1  = 83.4 #Walker
WEIGHT_2 = 97.4  #Williams
WEIGHT_3 = 109.0  #Battye
WEIGHT_4 = 78.6
WEIGHT_5 = 78.6
WEIGHT_6 = 78.6
WEIGHT_7 = 78.6

""" Length of play in minutes"""
PLAY_TIME = 40 # Minutes
#PLAY_TIME = 80 # Minutes

#%% MET data taken from:

# https://sites.google.com/site/compendiumofphysicalactivities/Activity-Categories/running
 
Mph         = np.array([4.0, 5.0, 5.2, 6.0, 6.7, 7.0, 8.0, 8.6, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
METS        = np.array([6.0, 8.3, 9.0, 9.8, 10.5, 11.0, 11.8, 12.3, 12.8, 14.5, 16.0, 19.0, 19.8, 23.0])

# Calculate velocity in meters per second
ms = Mph*(1.6/3.6)

# I'm going to fit a linear trend to the data -> so i can interpolate at the velocity midpoints
coefs = np.polynomial.polynomial.polyfit(ms, METS, 1)

#%% Calculations
"""
 Calories = METS x weight (kg) x time (hours)
 
 Joules = 4.184 x METS x weight (kg) x time (hours)
 
 I am going to assume that the distance covered in a speed band was at the mid velocity
 
 Speed band 1:      1.5 m/s
 Speed band 2:      4.5 m/s
 Speed band 3:      6.5 m/s
 Speed band 4:      8.5 m/s
 
I am also going to assume that the power was generated over 40 minutes
"""

# Mid-point velocities 
VELO = np.array([1.5, 4.5, 6.5, 8.5])

# These are the METS values at the mid-point velocities
METS_Vals = coefs[1]*VELO + coefs[0]

#%% Do the calculations

# Calculate the time spent in each speed band
TIMES_1 = SPEED_BANDS_1/VELO
TIMES_2 = SPEED_BANDS_2/VELO
TIMES_3 = SPEED_BANDS_3/VELO
TIMES_4 = SPEED_BANDS_4/VELO
TIMES_5 = SPEED_BANDS_5/VELO
TIMES_6 = SPEED_BANDS_6/VELO
TIMES_7 = SPEED_BANDS_7/VELO

# Calculate hours spent in each speed band
HOURS_1 = TIMES_1/3600.0
HOURS_2 = TIMES_2/3600.0
HOURS_3 = TIMES_3/3600.0
HOURS_4 = TIMES_4/3600.0
HOURS_5 = TIMES_5/3600.0
HOURS_6 = TIMES_6/3600.0
HOURS_7 = TIMES_7/3600.0

# Now calculate Joules (energy expenditure)
JOULES_1 = 1000*np.sum(4.184*METS_Vals*WEIGHT_1*HOURS_1) 
JOULES_2 = 1000*np.sum(4.184*METS_Vals*WEIGHT_2*HOURS_2)
JOULES_3 = 1000*np.sum(4.184*METS_Vals*WEIGHT_3*HOURS_3)
JOULES_4 = 1000*np.sum(4.184*METS_Vals*WEIGHT_4*HOURS_4)
JOULES_5 = 1000*np.sum(4.184*METS_Vals*WEIGHT_5*HOURS_5)
JOULES_6 = 1000*np.sum(4.184*METS_Vals*WEIGHT_6*HOURS_6)
JOULES_7 = 1000*np.sum(4.184*METS_Vals*WEIGHT_7*HOURS_7)

POWER_1 = JOULES_1/(PLAY_TIME*60)
POWER_2 = JOULES_2/(PLAY_TIME*60)
POWER_3 = JOULES_3/(PLAY_TIME*60)
POWER_4 = JOULES_4/(PLAY_TIME*60)
POWER_5 = JOULES_5/(PLAY_TIME*60)
POWER_6 = JOULES_6/(PLAY_TIME*60)
POWER_7 = JOULES_7/(PLAY_TIME*60)

print('\n\n\n\n')
print('Player 1 average power was:\t ', np.round(POWER_1,0),'Watts')
print('Player 2 average power was:\t ', np.round(POWER_2,0),'Watts')
print('Player 3 average power was:\t ', np.round(POWER_3,0),'Watts')
print('Player 4 average power was:\t ', np.round(POWER_4,0),'Watts')
print('Player 5 average power was:\t ', np.round(POWER_5,0),'Watts')
print('Player 6 average power was:\t ', np.round(POWER_6,0),'Watts')
print('Player 7 average power was:\t ', np.round(POWER_7,0),'Watts')

