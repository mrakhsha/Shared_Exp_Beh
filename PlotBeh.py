


"""
This script is to plot the behavioral analysis
Written by Mohsen Rakhshan at CCNL Dartmouth 20181002
"""
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple

import dill 
filename = 'Behavior20181220.pkl' 
dill.load_session(filename)
#%%
#################### plot performance separated by attend first ###############

PerformanceAll = BehResultAllSideAllRewAllConfAllAttend["Performance"]

MeanPerformanceAll = np.nanmean(PerformanceAll[PerformanceAll>0])
STDPerformanceAll  = np.nanstd(PerformanceAll[PerformanceAll>0])
SEMPerformanceAll  = stats.sem(PerformanceAll[PerformanceAll>0], axis=None, ddof=0)



PerformanceAttendFirst = BehResultAllSideAllRewAllConfAttendFirst["Performance"]

MeanPerformanceAttendFirst = np.nanmean(PerformanceAttendFirst[PerformanceAttendFirst>0])
STDPerformanceAttendFirst  = np.nanstd(PerformanceAttendFirst[PerformanceAttendFirst>0])
SEMPerformanceAttendFirst  = stats.sem(PerformanceAttendFirst[PerformanceAttendFirst>0], axis=None, ddof=0)

PerformanceAttendSec = BehResultAllSideAllRewAllConfAttendSec["Performance"]

MeanPerformanceAttendSec = np.nanmean(PerformanceAttendSec[PerformanceAttendSec>0])
STDPerformanceAttendSec  = np.nanstd(PerformanceAttendSec[PerformanceAttendSec>0])
SEMPerformanceAttendSec  = stats.sem(PerformanceAttendSec[PerformanceAttendSec>0], axis=None, ddof=0)

# check if attend first vs attend second is significant
TtestAttendFirstAttendSec = stats.ttest_rel(PerformanceAttendFirst[PerformanceAttendFirst>0],PerformanceAttendSec[PerformanceAttendSec>0])

# Take home message: There is no significant difference in performance

#%% Plot mean of each subject in two cases of attend first and attend second to see if there is two diffwerent type of people

fig = plt.figure(0)
x = np.arange(24)
y = np.nanmean(PerformanceAttendFirst,1)
yerr = np.nanstd(PerformanceAttendFirst,1)/np.sqrt(PerformanceAttendFirst.shape[1])
plt.errorbar(x, y, yerr=yerr)

y = np.nanmean(PerformanceAttendSec,1)
yerr = np.nanstd(PerformanceAttendSec,1)/np.sqrt(PerformanceAttendSec.shape[1])
plt.errorbar(x, y, yerr=yerr)

#%%
#################### plot RT separated by attend first ###############
SubjectRTAll = BehResultAllSideAllRewAllConfAllAttend["MeanSubjectRT"]



MeanSubjectRTAll = np.nanmean(SubjectRTAll[SubjectRTAll>0])
STDSubjectRTAll  = np.nanstd(SubjectRTAll[SubjectRTAll>0])
SEMSubjectRTAll  = stats.sem(SubjectRTAll[SubjectRTAll>0], axis=None, ddof=0)



SubjectRTAttendFirst = BehResultAllSideAllRewAllConfAttendFirst["MeanSubjectRT"]

MeanSubjectRTAttendFirst = np.nanmean(SubjectRTAttendFirst[SubjectRTAttendFirst>0])
STDSubjectRTAttendFirst  = np.nanstd(SubjectRTAttendFirst[SubjectRTAttendFirst>0])
SEMSubjectRTAttendFirst  = stats.sem(SubjectRTAttendFirst[SubjectRTAttendFirst>0], axis=None, ddof=0)

SubjectRTAttendSec = BehResultAllSideAllRewAllConfAttendSec["MeanSubjectRT"]

MeanSubjectRTAttendSec = np.nanmean(SubjectRTAttendSec[SubjectRTAttendSec>0])
STDSubjectRTAttendSec  = np.nanstd(SubjectRTAttendSec[SubjectRTAttendSec>0])
SEMSubjectRTAttendSec  = stats.sem(SubjectRTAttendSec[SubjectRTAttendSec>0], axis=None, ddof=0)


TtestAttendFirstSecond = stats.ttest_rel(SubjectRTAttendFirst[SubjectRTAttendFirst>0],SubjectRTAttendSec[SubjectRTAttendSec>0])

# Attend seconds are marginally faster than attend firsts (p=0.056)

#%%
#################### plot performance separated by High rew ##################

PerformanceAll = BehResultAllSideAllRewAllConfAllAttend["Performance"]

MeanPerformanceAll = np.nanmean(PerformanceAll[PerformanceAll>0])
STDPerformanceAll  = np.nanstd(PerformanceAll[PerformanceAll>0])
SEMPerformanceAll  = stats.sem(PerformanceAll[PerformanceAll>0], axis=None, ddof=0)


PerformanceHighRew = BehResultAllSideHighRewAllConfAllAttend["Performance"]

MeanPerformanceHighRew = np.nanmean(PerformanceHighRew[PerformanceHighRew>0])
STDPerformanceHighRew  = np.nanstd(PerformanceHighRew[PerformanceHighRew>0])
SEMPerformanceHighRew  = stats.sem(PerformanceHighRew[PerformanceHighRew>0], axis=None, ddof=0)

PerformanceLowRew = BehResultAllSideLowRewAllConfAllAttend["Performance"]

MeanPerformanceLowRew = np.nanmean(PerformanceLowRew[PerformanceLowRew>0])
STDPerformanceLowRew  = np.nanstd(PerformanceLowRew[PerformanceLowRew>0])
SEMPerformanceLowRew  = stats.sem(PerformanceLowRew[PerformanceLowRew>0], axis=None, ddof=0)

# check if attend first vs attend second is significant
TtestHighRewLowRew = stats.ttest_rel(PerformanceHighRew[PerformanceHighRew>0],PerformanceLowRew[PerformanceLowRew>0])

##############################################################################
#%% Plot mean of each subject in two cases of attend high reward and attend low reward to see if there is two diffwerent type of people

fig = plt.figure(0)
x = np.arange(24)
y = np.nanmean(PerformanceHighRew,1)
yerr = np.nanstd(PerformanceHighRew,1)/np.sqrt(PerformanceHighRew.shape[1])
plt.errorbar(x, y, yerr=yerr)

y = np.nanmean(PerformanceLowRew,1)
yerr = np.nanstd(PerformanceLowRew,1)/np.sqrt(PerformanceLowRew.shape[1])
plt.errorbar(x, y, yerr=yerr)


fig = plt.figure(1)
x = np.arange(24)
y = np.nanmean(PerformanceHighRew,1) - np.nanmean(PerformanceLowRew,1)
yerr = (np.nanstd(PerformanceHighRew,1)+np.nanstd(PerformanceLowRew,1))/np.sqrt(PerformanceHighRew.shape[1])
plt.errorbar(x, y, yerr=yerr)

#%%
#################### plot SubjectRT separated by High rew ##################

SubjectRTAll = BehResultAllSideAllRewAllConfAllAttend["MeanSubjectRT"]

MeanSubjectRTAll = np.nanmean(SubjectRTAll[SubjectRTAll>0])
STDSubjectRTAll  = np.nanstd(SubjectRTAll[SubjectRTAll>0])
SEMSubjectRTAll  = stats.sem(SubjectRTAll[SubjectRTAll>0], axis=None, ddof=0)


SubjectRTHighRew = BehResultAllSideHighRewAllConfAllAttend["MeanSubjectRT"]

MeanSubjectRTHighRew = np.nanmean(SubjectRTHighRew[SubjectRTHighRew>0])
STDSubjectRTHighRew  = np.nanstd(SubjectRTHighRew[SubjectRTHighRew>0])
SEMSubjectRTHighRew  = stats.sem(SubjectRTHighRew[SubjectRTHighRew>0], axis=None, ddof=0)

SubjectRTLowRew = BehResultAllSideLowRewAllConfAllAttend["MeanSubjectRT"]

MeanSubjectRTLowRew = np.nanmean(SubjectRTLowRew[SubjectRTLowRew>0])
STDSubjectRTLowRew  = np.nanstd(SubjectRTLowRew[SubjectRTLowRew>0])
SEMSubjectRTLowRew  = stats.sem(SubjectRTLowRew[SubjectRTLowRew>0], axis=None, ddof=0)

# check if attend first vs attend second is significant
TtestHighRewLowRew = stats.ttest_rel(SubjectRTHighRew[SubjectRTHighRew>0],SubjectRTLowRew[SubjectRTLowRew>0])

###############################################################################
# Take home message: High rewards are significantly faster (p=0.00016)
# Low rew mean:0.5438997028143325
# high rew mean: 0.5249752566487803

#%%
#################### plot performance separated by High conf ##################

PerformanceAll = BehResultAllSideAllRewAllConfAllAttend["Performance"]

MeanPerformanceAll = np.nanmean(PerformanceAll[PerformanceAll>0])
STDPerformanceAll  = np.nanstd(PerformanceAll[PerformanceAll>0])
SEMPerformanceAll  = stats.sem(PerformanceAll[PerformanceAll>0], axis=None, ddof=0)


PerformanceHighConf = BehResultAllSideAllRewHighConfAllAttend["Performance"]

MeanPerformanceHighConf = np.nanmean(PerformanceHighConf[PerformanceHighConf>0])
STDPerformanceHighConf  = np.nanstd(PerformanceHighConf[PerformanceHighConf>0])
SEMPerformanceHighConf  = stats.sem(PerformanceHighConf[PerformanceHighConf>0], axis=None, ddof=0)

PerformanceLowConf = BehResultAllSideAllRewLowConfAllAttend["Performance"]

MeanPerformanceLowConf = np.nanmean(PerformanceLowConf[PerformanceLowConf>0])
STDPerformanceLowConf  = np.nanstd(PerformanceLowConf[PerformanceLowConf>0])
SEMPerformanceLowConf  = stats.sem(PerformanceLowConf[PerformanceLowConf>0], axis=None, ddof=0,nan_policy='omit')

# check if attend first vs attend second is significant
RanksumHighRewLowRew = stats.ranksums(PerformanceHighConf[PerformanceHighConf>0],PerformanceLowConf[PerformanceLowConf>0])

##############################################################################
#%% Plot mean of each subject in two cases of attend high reward and attend low reward to see if there is two diffwerent type of people

fig = plt.figure(0)
x = np.arange(24)
y = np.nanmean(PerformanceHighConf,1)
yerr = np.nanstd(PerformanceHighConf,1)/np.sqrt(PerformanceHighConf.shape[1])
plt.errorbar(x, y, yerr=yerr)

y = np.nanmean(PerformanceLowConf,1)
yerr = np.nanstd(PerformanceLowConf,1)/np.sqrt(PerformanceLowConf.shape[1])
plt.errorbar(x, y, yerr=yerr)


fig = plt.figure(1)
x = np.arange(24)
y = np.nanmean(PerformanceHighConf,1) - np.nanmean(PerformanceLowConf,1)
yerr = (np.nanstd(PerformanceHighConf,1)+np.nanstd(PerformanceLowConf,1))/np.sqrt(PerformanceLowConf.shape[1])
plt.errorbar(x, y, yerr=yerr)
#%%
##################### plot SubjectRT separated by High conf ##################

SubjectRTAll = BehResultAllSideAllRewAllConfAllAttend["MeanSubjectRT"]

MeanSubjectRTAll = np.nanmean(SubjectRTAll[SubjectRTAll>0])
STDSubjectRTAll  = np.nanstd(SubjectRTAll[SubjectRTAll>0])
SEMSubjectRTAll  = stats.sem(SubjectRTAll[SubjectRTAll>0], axis=None, ddof=0)


SubjectRTHighConf = BehResultAllSideAllRewHighConfAllAttend["MeanSubjectRT"]

MeanSubjectRTHighConf = np.nanmean(SubjectRTHighConf[SubjectRTHighConf>0])
STDSubjectRTHighConf  = np.nanstd(SubjectRTHighConf[SubjectRTHighConf>0])
SEMSubjectRTHighConf  = stats.sem(SubjectRTHighConf[SubjectRTHighConf>0], axis=None, ddof=0)

SubjectRTLowConf = BehResultAllSideAllRewLowConfAllAttend["MeanSubjectRT"]

MeanSubjectRTLowConf = np.nanmean(SubjectRTLowConf[SubjectRTLowConf>0])
STDSubjectRTLowConf  = np.nanstd(SubjectRTLowConf[SubjectRTLowConf>0])
SEMSubjectRTLowConf  = stats.sem(SubjectRTLowConf[SubjectRTLowConf>0], axis=None, ddof=0,nan_policy='omit')

# check if attend first vs attend second is significant
RanksumHighRewLowRew = stats.ranksums(SubjectRTHighConf[SubjectRTHighConf>0],SubjectRTLowConf[SubjectRTLowConf>0])


