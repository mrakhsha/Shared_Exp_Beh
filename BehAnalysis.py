"""
This code is written for behavioral analysis of the EEG shared experiment
Written by Mohsen Rakhshan 

To do list: I should calculate WinStay LoseSwitch and the others on the real indices and not in the idx_selective
"""
# import necessary libraries and packages
import scipy.io
import numpy as np
# import pdb
import dill
from scipy import stats

# Directory of the behavioral data
file_directory = '/Users/mohsen/Dropbox (CCNL-Dartmouth)/Shared Experiments REWARD/beh_data/'
# list of subjects
subject_list = ['behav_Shared_ARSubNum21', 'behav_Shared_ESSubNum24', 'behav_Shared_HASubNum20',
                'behav_Shared_JHSubNum29',
                'behav_Shared_JSSubNum25', 'behav_Shared_PDSubNum28', 'behav_Shared_SPSubNum27',
                'behav_Shared_STSubNum26',
                'behav_Shared_TLSubNum22', 'behav_Shared_TWSubNum30', 'behav_Shared_TZSubNum23',
                'behav_Shared_AHSubNum12',
                'behav_Shared_AMSubNum16', 'behav_Shared_ASSubNum18', 'behav_Shared_BJSSubNum14',
                'behav_Shared_BSSubNum15',
                'behav_Shared_JEVSubNum11', 'behav_Shared_JGSubNum19', 'behav_Shared_JSSubNum16',
                'behav_Shared_MHSubNum17',
                'behav_Shared_OKSubNum13', 'behav_Shared_GFSubNum15', 'behav_Shared_KKSubNum14',
                'behav_Shared_NH2SubNum13']


# subject_list  = ['Shared_GFSubNum15']
# subject_list  = ['Shared_GFSubNum15','Shared_KKSubNum14','Shared_NH2SubNum13']

# define the function which extracts the behavioral variables we need
def var_extractor(file_directory, subject_list):
    """
    
    :param file_directory: A string for the address of the input data
    :param subject_list: A list of inputs
    :return: a dictionary of all the variables of the experiment and behavior
    """
    beh_vars = []
    # main loop for loading the data
    for subject in range(len(subject_list)):
        beh_data = scipy.io.loadmat(file_directory + subject_list[subject] + '.mat')
        num_tar_att = np.array(beh_data["num_tar_att"])
        att_side = np.array(beh_data["att_side"])
        att_side = np.transpose(att_side)
        conf_val = np.array(beh_data["conf_val"])
        get_rew = np.array(beh_data["get_rew"])
        rew_val = np.array(beh_data["rew_val"])
        att_first = np.array(beh_data["att_first"])
        sub_rt = np.array(beh_data["sub_rt"])

        # Extracting the ones with only one target
        idx_one_tar = num_tar_att == 1
        tmp_conf_val = conf_val[idx_one_tar]
        tmp_get_rew = get_rew[idx_one_tar]
        tmp_rew_val = rew_val[idx_one_tar]
        tmp_sub_rt = sub_rt[idx_one_tar]
        tmp_att_first = att_first[idx_one_tar]
        # reshape to the block form
        conf_val = tmp_conf_val.reshape(att_side.shape[0], int(tmp_conf_val.shape[0] / att_side.shape[0]))
        get_rew = tmp_get_rew.reshape(att_side.shape[0], int(tmp_get_rew.shape[0] / att_side.shape[0]))
        rew_val = tmp_rew_val.reshape(att_side.shape[0], int(tmp_rew_val.shape[0] / att_side.shape[0]))
        sub_rt = tmp_sub_rt.reshape(att_side.shape[0], int(tmp_sub_rt.shape[0] / att_side.shape[0]))
        att_first = tmp_att_first.reshape(att_side.shape[0], int(tmp_att_first.shape[0] / att_side.shape[0]))
        # make a dictionary for necessary variables
        tmp_beh_vars = {"att_side": att_side, "conf_val": conf_val, "get_rew": get_rew, "rew_val": rew_val,
                        "sub_rt": sub_rt, "att_first": att_first}
        # append counters data together
        beh_vars.append(tmp_beh_vars)

    return beh_vars


# define a class of functions for behavior analysis
class Behavior:
    ...

    @staticmethod
    # calculating correct
    def performance(cor_vec):
        """
        
        :param cor_vec: numpy array of correct/incorrect
        :return: scalar of performance
        """
        mean_perf = (np.nanmean(cor_vec)) * 100
        return mean_perf

    # calculating probability of stay
    def prob_stay(cor_vec):
        pre_cor_vec = np.insert(cor_vec[:-1], 0, 0)
        idx_stay = np.array(pre_cor_vec == cor_vec)
        prob_stay = np.mean(idx_stay)
        return prob_stay

    # calculating probability of WinStay
    def prob_winstay(cor_vec):
        pre_cor_vec = np.insert(cor_vec[:-1], 0, 0)
        idx_stay = np.array(pre_cor_vec == cor_vec)
        prob_winstay = np.mean(idx_stay & pre_cor_vec) / np.mean(pre_cor_vec)
        return prob_winstay

    # calculating probability of LoseSwitch
    def prob_loseswitch(cor_vec):
        pre_cor_vec = np.insert(cor_vec[:-1], 0, 0)
        idx_switch = np.array(pre_cor_vec != cor_vec)
        pre_false_vec = ~(pre_cor_vec.astype(bool)) * 1
        prob_loseswitch = np.mean(idx_switch & pre_false_vec) / np.mean(pre_false_vec)
        return prob_loseswitch


def beh_analysis(beh_vars, idx_rew, idx_conf, idx_side, idx_att_first):
    if idx_side == 1 or idx_side == 2:

        num_block = int((beh_vars[0]["att_side"].shape[0]) / 2)
    else:
        num_block = int(beh_vars[0]["att_side"].shape[0])

    # initialization
    performance = np.nan * np.zeros(shape=(len(beh_vars), num_block))
    prob_stay = np.nan * np.zeros(shape=(len(beh_vars), num_block))
    prob_winstay = np.nan * np.zeros(shape=(len(beh_vars), num_block))
    prob_loseswitch = np.nan * np.zeros(shape=(len(beh_vars), num_block))
    sub_rt = []
    mean_sub_rt = np.nan * np.zeros(shape=(len(beh_vars), num_block))

    for subject in range(len(beh_vars)):

        tmp_beh_data = beh_vars[subject]
        tmp_beh_data1 = {}

        if idx_side == 1 or idx_side == 2:

            idx_side = np.where(beh_vars[0]["att_side"] == idx_side)[0]

            for key in tmp_beh_data.keys():
                tmp_beh_data1[key] = tmp_beh_data[key][idx_side, :]
        else:
            tmp_beh_data1 = tmp_beh_data

        for block in range(num_block):
            # calculate the average of correct over reward and confidence conditions
            if (idx_rew == 1 or idx_rew == 3) and (idx_conf != 2 and idx_conf != 1) and (
                    idx_att_first != 1 and idx_att_first != 0):
                idx_selective = np.where(tmp_beh_data1["rew_val"][block, :] == idx_rew)

            elif (idx_rew != 1 and idx_rew != 3) and (idx_conf == 2 or idx_conf == 1) and (
                    idx_att_first != 1 and idx_att_first != 0):
                idx_selective = np.where(tmp_beh_data1["conf_val"][block, :] == idx_conf)

            elif (idx_rew != 1 and idx_rew != 3) and (idx_conf != 2 and idx_conf != 1) and (
                    idx_att_first == 1 or idx_att_first == 0):
                idx_selective = np.where(tmp_beh_data1["att_first"][block, :] == idx_att_first)

            elif (idx_rew == 1 or idx_rew == 3) and (idx_conf == 2 or idx_conf == 1) and (
                    idx_att_first != 1 and idx_att_first != 0):
                idx_selective = np.intersect1d(np.where(tmp_beh_data1["conf_val"][block, :] == idx_conf), 
                                               np.where(tmp_beh_data1["rew_val"][block, :] == idx_rew))

            elif (idx_rew == 1 or idx_rew == 3) and (idx_conf != 2 and idx_conf != 1) and (
                    idx_att_first == 1 or idx_att_first == 0):
                idx_selective = np.intersect1d(np.where(tmp_beh_data1["rew_val"][block, :] == idx_rew), 
                                               np.where(tmp_beh_data1["att_first"][block, :] == idx_att_first))

            elif (idx_rew != 1 and idx_rew != 3) and (idx_conf == 2 or idx_conf == 1) and (
                    idx_att_first == 1 or idx_att_first == 0):
                idx_selective = np.intersect1d(np.where(tmp_beh_data1["conf_val"][block, :] == idx_conf), 
                                               np.where(tmp_beh_data1["att_first"][block, :] == idx_att_first))

            elif (idx_rew == 1 or idx_rew == 3) and (idx_conf == 2 or idx_conf == 1) and (
                    idx_att_first == 1 or idx_att_first == 0):
                idx_selective = reduce(np.intersect1d, (np.where(tmp_beh_data1["conf_val"][block, :] == idx_conf), 
                                                        np.where(tmp_beh_data1["rew_val"][block, :] == idx_rew), 
                                                        np.where(tmp_beh_data1["att_first"][block, :] == idx_att_first))
                                       )
            else:

                idx_selective = range(len(tmp_beh_data1["rew_val"][block, :]))

            cor_vec = tmp_beh_data1["get_rew"][block, idx_selective]
            performance[subject, block] = Behavior.performance(cor_vec)
            prob_stay[subject, block] = Behavior.prob_stay(cor_vec)
            prob_winstay[subject, block] = Behavior.prob_winstay(cor_vec)
            prob_loseswitch[subject, block] = Behavior.prob_loseswitch(cor_vec)
            tmp_rt = tmp_beh_data1["sub_rt"][block, idx_selective][cor_vec.astype(bool)]
            tmp_rt = tmp_rt[tmp_rt > 0]  # remove the ones which was no answer or negative
            sub_rt.append(tmp_rt)
            mean_sub_rt[subject, block] = np.mean(tmp_rt)

        beh_result = {"performance": performance, "prob_stay": prob_stay, "prob_winstay": prob_winstay, 
                      "prob_loseswitch": prob_loseswitch, "sub_rt": sub_rt, "mean_sub_rt": mean_sub_rt}

    return beh_result


# Extracting the behavioral variables #############################
beh_vars = var_extractor(file_directory, subject_list)

# Instruction ##############################

# idx_side = 0 means all the trials
# idx_side = 1 means the trials of side 1
# idx_side = 2 means the trials of side 2

# idx_rew = 0 means all the trials
# idx_rew = 3 means the trials with reward 3
# idx_rew = 1 means the trials with reward 1

# idx_conf = 0 means all the trials
# idx_conf = 2 means the trials with confidence 2
# idx_conf = 1 means the trials with confidence 1

# idx_att_first = 2
# idx_att_first = 1
# idx_att_first = 0

# Strategy over all trials #############################
idx_rew = 0
idx_conf = 0
idx_side = 0
idx_att_first = 2

beh0002 = beh_analysis(beh_vars, idx_rew, idx_conf, idx_side, idx_att_first)

# Strategy over all trials #############################
idx_rew = 0
idx_conf = 0
idx_side = 1
idx_att_first = 2

beh0012 = beh_analysis(beh_vars, idx_rew, idx_conf, idx_side, idx_att_first)

#
idx_rew = 0
idx_conf = 0
idx_side = 2
idx_att_first = 2

beh0022 = beh_analysis(beh_vars, idx_rew, idx_conf, idx_side, idx_att_first)

perf1 = beh0012["performance"]
perf2 = beh0022["performance"]

mean_perf1 = np.mean(perf1, 1)
mean_perf2 = np.mean(perf2, 1)
ttest_mean12 = stats.ttest_rel(np.delete(mean_perf1, 18), np.delete(mean_perf2, 18))

perf1 = perf1.reshape(perf1.shape[0] * perf1.shape[1], 1)
perf2 = perf2.reshape(perf2.shape[0] * perf2.shape[1], 1)
ttest_12 = stats.ttest_rel(perf1, perf2)
ranksum_12 = stats.ranksums(perf1, perf2)

# %%
# Strategy over all trials attend first ##########################
idx_rew = 0
idx_conf = 0
idx_side = 0
idx_att_first = 1

beh0001 = beh_analysis(beh_vars, idx_rew, idx_conf, idx_side, idx_att_first)

# %%
# Strategy over all trials attend second #########################
idx_rew = 0
idx_conf = 0
idx_side = 0
idx_att_first = 0

beh0000 = beh_analysis(beh_vars, idx_rew, idx_conf, idx_side, idx_att_first)

# %%
# Strategy over all trials highreward #############################
idx_rew = 3
idx_conf = 0
idx_side = 0
idx_att_first = 2

beh3002 = beh_analysis(beh_vars, idx_rew, idx_conf, idx_side, idx_att_first)
# %%
# Strategy over all trials lowreward #############################
idx_rew = 1
idx_conf = 0
idx_side = 0
idx_att_first = 2

beh1002 = beh_analysis(beh_vars, idx_rew, idx_conf, idx_side, idx_att_first)
# %%
# Strategy over all trials highconfidence #######################
idx_rew = 0
idx_conf = 2
idx_side = 0
idx_att_first = 2

beh0202 = beh_analysis(beh_vars, idx_rew, idx_conf, idx_side, idx_att_first)
# %%
# Strategy over all trials lowconfidence #########################
idx_rew = 0
idx_conf = 1
idx_side = 0
idx_att_first = 2

beh0102 = beh_analysis(beh_vars, idx_rew, idx_conf, idx_side, idx_att_first)

#  Saving the variables ########################
# %%
# pip install dill --user
filename = 'Behavior20181220.pkl'

dill.dump_session(filename)

# and to load the session again:
# import dill
# filename = 'Behavior.pkl'
# dill.load_session(filename)
