
import Shared_Exp_Beh as seb
import os.path as op
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

data_path = op.join(seb.__path__[0], 'data/')
file_directory = data_path
fig_directory = op.join(seb.__path__[0], 'figures/')

subject_list = ['behav_Shared_ARSubNum21', 'behav_Shared_ESSubNum24', 'behav_Shared_HASubNum20',
                'behav_Shared_JHSubNum29',
                'behav_Shared_JSSubNum25', 'behav_Shared_PDSubNum28', 'behav_Shared_SPSubNum27',
                'behav_Shared_STSubNum26',
                'behav_Shared_TLSubNum22', 'behav_Shared_TWSubNum30', 'behav_Shared_TZSubNum23',
                'behav_Shared_AHSubNum12',
                'behav_Shared_ASSubNum18', 'behav_Shared_BJSSubNum14',
                'behav_Shared_BSSubNum15',
                'behav_Shared_JEVSubNum11', 'behav_Shared_JGSubNum19', 'behav_Shared_JSSubNum16',
                'behav_Shared_MHSubNum17',
                'behav_Shared_OKSubNum13']

beh_vars = seb.var_extractor(file_directory, subject_list)


# idx_side = 0 means all the trials
# idx_side = 1 means the trials of side 1
# idx_side = 2 means the trials of side 2

# idx_rew = 0 means all the trials
# idx_rew = 3 means the trials with reward 3
# idx_rew = 1 means the trials with reward 1

# idx_conf = 0 means all the trials
# idx_conf = 2 means the trials with confidence 2
# idx_conf = 1 means the trials with confidence 1

# idx_att_first = 2 means all trials
# idx_att_first = 1 means trials that target appeared earlier in the attended stream
# idx_att_first = 0 means trials that target appeared later in the attended stream


# looking only different rewards
mean_perf = []
sem_perf = []
mean_rt = []
sem_rt = []

for id_rew in [0, 1, 3]:

    beh = seb.beh_analysis(beh_vars, idx_rew=id_rew, idx_conf=0, idx_side=0, idx_att_first=2)
    avgperf = np.nanmean(beh["performance"])
    semperf = stats.sem(beh["performance"], axis=None, ddof=0, nan_policy='omit')
    avgrt = np.nanmean(beh["mean_sub_rt"])
    semrt = stats.sem(beh["mean_sub_rt"], axis=None, ddof=0, nan_policy='omit')
    mean_perf.append(avgperf)
    sem_perf.append(semperf)
    mean_rt.append(avgrt)
    sem_rt.append(semrt)

# plotting all the performances in different reward conditions
fig = plt.figure(1)
plt.errorbar(np.arange(3), mean_perf, yerr=sem_perf)
ax = fig.gca()
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('Accuracy', fontsize=20)
ax.set_xticks(range(3))
ax.set_xticklabels(('All', 'LowRew', 'HighRew'))
for axis in ['bottom', 'left']:
  ax.spines[axis].set_linewidth(3)
ax.tick_params(width=3)
ax.set_ylim([74, 81])
fig.tight_layout()
plt.yticks(np.arange(74, 81, 1))
plt.tick_params(labelsize=15)
plt.show()
fig.savefig(fig_directory+"AccuracyRew.pdf")

# plotting all the rt in different reward conditions
fig = plt.figure(2)
plt.errorbar(np.arange(3), mean_rt, yerr=sem_rt)
ax = fig.gca()
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('z(RT)', fontsize=20)
ax.set_xticks(range(3))
ax.set_xticklabels(('All', 'LowRew', 'HighRew'))
for axis in ['bottom', 'left']:
  ax.spines[axis].set_linewidth(3)
ax.tick_params(width=3)
ax.set_ylim([0.52, 0.57])
fig.tight_layout()
plt.yticks(np.arange(0.52, 0.57, 0.01))
plt.tick_params(labelsize=15)
plt.show()
fig.savefig(fig_directory+"RTRew.pdf")

# looking only different confidences
mean_perf = []
sem_perf = []
mean_rt = []
sem_rt = []

for idx_conf in [0, 1, 2]:

    beh = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=idx_conf, idx_side=0, idx_att_first=2)
    avgperf = np.nanmean(beh["performance"])
    semperf = stats.sem(beh["performance"], axis=None, ddof=0, nan_policy='omit')
    avgrt = np.nanmean(beh["mean_sub_rt"])
    semrt = stats.sem(beh["mean_sub_rt"], axis=None, ddof=0, nan_policy='omit')
    mean_perf.append(avgperf)
    sem_perf.append(semperf)
    mean_rt.append(avgrt)
    sem_rt.append(semrt)

# plotting all the performances in different confidence conditions
fig = plt.figure(1)
plt.errorbar(np.arange(3), mean_perf, yerr=sem_perf)
ax = fig.gca()
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('Accuracy', fontsize=20)
ax.set_xticks(range(3))
ax.set_xticklabels(('All', 'LowConf', 'HighConf'))
for axis in ['bottom', 'left']:
  ax.spines[axis].set_linewidth(3)
ax.tick_params(width=3)
ax.set_ylim([74, 81])
fig.tight_layout()
plt.yticks(np.arange(74, 81, 1))
plt.tick_params(labelsize=15)
plt.show()
fig.savefig(fig_directory+"AccuracyConf.pdf")

# plotting all the rt in different confidence conditions
fig = plt.figure(2)
plt.errorbar(np.arange(3), mean_rt, yerr=sem_rt)
ax = fig.gca()
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('z(RT)', fontsize=20)
ax.set_xticks(range(3))
ax.set_xticklabels(('All', 'LowConf', 'HighConf'))
for axis in ['bottom', 'left']:
  ax.spines[axis].set_linewidth(3)
ax.tick_params(width=3)
ax.set_ylim([0.52, 0.57])
fig.tight_layout()
plt.yticks(np.arange(0.52, 0.57, 0.01))
plt.tick_params(labelsize=15)
plt.show()
fig.savefig(fig_directory+"RTConf.pdf")


# looking only different side of attention
mean_perf = []
sem_perf = []
mean_rt = []
sem_rt = []

for idx_side in [0, 1, 2]:

    beh = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=0, idx_side=idx_side, idx_att_first=2)
    avgperf = np.nanmean(beh["performance"])
    semperf = stats.sem(beh["performance"], axis=None, ddof=0, nan_policy='omit')
    avgrt = np.nanmean(beh["mean_sub_rt"])
    semrt = stats.sem(beh["mean_sub_rt"], axis=None, ddof=0, nan_policy='omit')
    mean_perf.append(avgperf)
    sem_perf.append(semperf)
    mean_rt.append(avgrt)
    sem_rt.append(semrt)

# plotting all the performances in different confidence conditions
fig = plt.figure(1)
plt.errorbar(np.arange(3), mean_perf, yerr=sem_perf)
ax = fig.gca()
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('Accuracy', fontsize=20)
ax.set_xticks(range(3))
ax.set_xticklabels(('All', 'Left', 'Right'))
for axis in ['bottom', 'left']:
  ax.spines[axis].set_linewidth(3)
ax.tick_params(width=3)
ax.set_ylim([74, 81])
fig.tight_layout()
plt.yticks(np.arange(74, 81, 1))
plt.tick_params(labelsize=15)
plt.show()
fig.savefig(fig_directory+"AccuracySide.pdf")

# plotting all the rt in different confidence conditions
fig = plt.figure(2)
plt.errorbar(np.arange(3), mean_rt, yerr=sem_rt)
ax = fig.gca()
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('z(RT)', fontsize=20)
ax.set_xticks(range(3))
ax.set_xticklabels(('All', 'Left', 'Right'))
for axis in ['bottom', 'left']:
  ax.spines[axis].set_linewidth(3)
ax.tick_params(width=3)
ax.set_ylim([0.52, 0.57])
fig.tight_layout()
plt.yticks(np.arange(0.52, 0.57, 0.01))
plt.tick_params(labelsize=15)
plt.show()
fig.savefig(fig_directory+"RTSide.pdf")


# looking only different order of attention
mean_perf = []
sem_perf = []
mean_rt = []
sem_rt = []

for idx_att_first in [2, 1, 0]:

    beh = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=0, idx_side=0, idx_att_first=idx_att_first)
    avgperf = np.nanmean(beh["performance"])
    semperf = stats.sem(beh["performance"], axis=None, ddof=0, nan_policy='omit')
    avgrt = np.nanmean(beh["mean_sub_rt"])
    semrt = stats.sem(beh["mean_sub_rt"], axis=None, ddof=0, nan_policy='omit')
    mean_perf.append(avgperf)
    sem_perf.append(semperf)
    mean_rt.append(avgrt)
    sem_rt.append(semrt)

# plotting all the performances in different confidence conditions
fig = plt.figure(1)
plt.errorbar(np.arange(3), mean_perf, yerr=sem_perf)
ax = fig.gca()
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('Accuracy', fontsize=20)
ax.set_xticks(range(3))
ax.set_xticklabels(('All', 'AttFirst', 'AttSecond'))
for axis in ['bottom', 'left']:
  ax.spines[axis].set_linewidth(3)
ax.tick_params(width=3)
ax.set_ylim([74, 81])
fig.tight_layout()
plt.yticks(np.arange(74, 81, 1))
plt.tick_params(labelsize=15)
plt.show()
fig.savefig(fig_directory+"AccuracyOrder.pdf")

# plotting all the rt in different confidence conditions
fig = plt.figure(2)
plt.errorbar(np.arange(3), mean_rt, yerr=sem_rt)
ax = fig.gca()
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('z(RT)', fontsize=20)
ax.set_xticks(range(3))
ax.set_xticklabels(('All', 'AttFirst', 'AttSecond'))
for axis in ['bottom', 'left']:
  ax.spines[axis].set_linewidth(3)
ax.tick_params(width=3)
ax.set_ylim([0.52, 0.57])
fig.tight_layout()
plt.yticks(np.arange(0.52, 0.57, 0.01))
plt.tick_params(labelsize=15)
plt.show()
fig.savefig(fig_directory+"RTOrder.pdf")
