# # idx_side = 0 means all the trials
# # idx_side = 1 means the trials of side 1
# # idx_side = 2 means the trials of side 2
#
# # idx_rew = 0 means all the trials
# # idx_rew = 3 means the trials with reward 3
# # idx_rew = 1 means the trials with reward 1
#
# # idx_conf = 0 means all the trials
# # idx_conf = 2 means the trials with confidence 2
# # idx_conf = 1 means the trials with confidence 1
#
# # idx_att_first = 2 means all trials
# # idx_att_first = 1 means trials that target appeared earlier in the attended stream
# # idx_att_first = 0 means trials that target appeared later in the attended stream
#
# beh0002 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=0, idx_side=0, idx_att_first=2)
# beh0012 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=0, idx_side=1, idx_att_first=2)
# beh0022 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=0, idx_side=2, idx_att_first=2)
# beh0001 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=0, idx_side=0, idx_att_first=1)
# beh0000 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=0, idx_side=0, idx_att_first=0)
# beh3002 = seb.beh_analysis(beh_vars, idx_rew=3, idx_conf=0, idx_side=0, idx_att_first=2)
# beh1002 = seb.beh_analysis(beh_vars, idx_rew=1, idx_conf=0, idx_side=0, idx_att_first=2)
# beh0202 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=2, idx_side=0, idx_att_first=2)
# beh0102 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=1, idx_side=0, idx_att_first=2)
#
# beh_all = (beh0002, beh0012, beh0022, beh0001, beh0000, beh3002, beh1002, beh0202, beh0102)

from scipy import stats
import numpy as np
# import matplotlib.pyplot as plt
# Plot mean of each subject in two cases of attend first and attend second (if there are two different type of people)
# fig = plt.figure(0)
# x = np.arange(24)
# y = np.nanmean(beh0001["performance"], 1)
# yerr = np.nanstd(beh0001["performance"], 1) / np.sqrt(beh0001["performance"].shape[1])
# plt.errorbar(x, y, yerr=yerr)
#
# y = np.nanmean(beh0000["performance"], 1)
# yerr = np.nanstd(beh0000["performance"], 1) / np.sqrt(beh0000["performance"].shape[1])
# plt.errorbar(x, y, yerr=yerr)
#
# # check if the difference between the performance when target comes earlier or later in attended stream is different
# print(stats.ttest_rel(beh0001["performance"].flatten(), beh0000["performance"].flatten()))
#
# # check the difference between the performance high rew vs low rew
# print(stats.ttest_rel(beh3002["performance"].flatten(), beh1002["performance"].flatten()))
#
# # check the difference between the performance high conf vs low conf (it is significant)
# id_valid = np.logical_and(beh0102["performance"].flatten()>0, beh0202["performance"].flatten()>0)
# print(stats.ttest_rel(beh0202["performance"].flatten()[id_valid], beh0102["performance"].flatten()[id_valid]))
#
# # check if the difference between the performance for different sides (side bias)
# print(stats.ttest_rel(beh0022["performance"].flatten(), beh0012["performance"].flatten()))
