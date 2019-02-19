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


# define a class of functions for behavior analysis
class Behavior:
    """
            Blueprint for behavior

    """

    def __init__(self, cor_vec):
        self.cor_vec = cor_vec

    # calculating performance
    def performance(self):

        mean_perf = (np.nanmean(self.cor_vec)) * 100
        return mean_perf

    # calculating probability of stay
    def prob_stay(self):

        pre_cor_vec = np.insert(self.cor_vec[:-1], 0, 0)
        idx_stay = np.array(pre_cor_vec == self.cor_vec)
        prob_stay = np.mean(idx_stay)
        return prob_stay

    # calculating probability of WinStay
    def prob_winstay(self):
        pre_cor_vec = np.insert(self.cor_vec[:-1], 0, 0)
        idx_stay = np.array(pre_cor_vec == self.cor_vec)
        prob_winstay = np.mean(idx_stay & pre_cor_vec) / np.mean(pre_cor_vec)
        return prob_winstay

    # calculating probability of LoseSwitch
    def prob_loseswitch(self):
        pre_cor_vec = np.insert(self.cor_vec[:-1], 0, 0)
        idx_switch = np.array(pre_cor_vec != self.cor_vec)
        pre_false_vec = ~(pre_cor_vec.astype(bool)) * 1
        prob_loseswitch = np.mean(idx_switch & pre_false_vec) / np.mean(pre_false_vec)
        return prob_loseswitch


# define the function which extracts the behavioral variables we need
def var_extractor(file_directory, subject_list):
    """
    
    :param file_directory: A string for the address of the input data
    :param subject_list: A list of inputs
    :return: a dictionary of all the variables of the experiment and behavior
    """

    beh_vars = []

    # main loop for loading the data
    for subject in subject_list:
        beh_data = scipy.io.loadmat(file_directory + subject + '.mat')
        num_tar_att = np.array(beh_data["NumTargetAttended"])
        att_side = np.array(beh_data["AttendedSide"])
        att_side = np.transpose(att_side)
        conf_val = np.array(beh_data["ConfidenceValue"])
        get_rew = np.array(beh_data["GetReward"])
        rew_val = np.array(beh_data["RewardVecValue"])
        att_first = np.array(beh_data["AttendedFirst"])
        sub_rt = np.array(beh_data["SubjectRT"])

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


def beh_analysis(beh_vars, idx_rew, idx_conf, idx_side, idx_att_first):
    """

    :param beh_vars: a dictionary of the inputs of the behavioral parameters
    :param idx_rew: int to show which reward value we need
    :param idx_conf: int to show which confidence results we want
    :param idx_side: int to show which side we want
    :param idx_att_first: int shows whether we want the trials in which target appears in attended stream earlier
    :return:
            a dictionary of all behavioral policies
    """

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

    for sub_beh in beh_vars:

        tmp_beh_data = sub_beh
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


def main():
    # Directory of the behavioral data (To do: Change this to argument)
    file_directory = '/Users/mohsen/Dropbox (CCNL-Dartmouth)/Shared Experiments REWARD/beh_data/'
    # list of subjects (To do: Change this to argument)
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

    beh0002 = beh_analysis(beh_vars, idx_rew=0, idx_conf=0, idx_side=0, idx_att_first=2)
    beh0012 = beh_analysis(beh_vars, idx_rew=0, idx_conf=0, idx_side=1, idx_att_first=2)
    beh0022 = beh_analysis(beh_vars, idx_rew=0, idx_conf=0, idx_side=2, idx_att_first=2)
    beh0001 = beh_analysis(beh_vars, idx_rew=0, idx_conf=0, idx_side=0, idx_att_first=1)
    beh0000 = beh_analysis(beh_vars, idx_rew=0, idx_conf=0, idx_side=0, idx_att_first=0)
    beh3002 = beh_analysis(beh_vars, idx_rew=3, idx_conf=0, idx_side=0, idx_att_first=2)
    beh1002 = beh_analysis(beh_vars, idx_rew=1, idx_conf=0, idx_side=0, idx_att_first=2)
    beh0202 = beh_analysis(beh_vars, idx_rew=0, idx_conf=2, idx_side=0, idx_att_first=2)
    beh0102 = beh_analysis(beh_vars, idx_rew=0, idx_conf=1, idx_side=0, idx_att_first=2)

    beh_all = (beh0002, beh0012, beh0022, beh0001, beh0000, beh3002, beh1002, beh0202, beh0102)

    return beh_all


if __name__ == '__main__':
    main()
