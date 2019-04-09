"""
This code is written for behavioral analysis of the EEG shared experiment
Written by Mohsen Rakhshan 

"""
# import necessary libraries and packages
import scipy.io
import numpy as np


# define a class of functions for behavior analysis
class Behavior:
    """
            Blueprint for behavior

    """

    @staticmethod
    # calculating performance
    def performance(cor_vec):

        mean_perf = (np.nanmean(cor_vec)) * 100
        return mean_perf

    @staticmethod
    # calculating probability of stay
    def prob_stay(cor_vec, pre_cor_vec):

        idx_stay = np.array(pre_cor_vec == cor_vec)
        prob_stay = np.mean(idx_stay)
        return prob_stay

    @staticmethod
    # calculating probability of WinStay
    def prob_winstay(cor_vec, pre_cor_vec):

        idx_stay = np.array(pre_cor_vec == cor_vec)
        if np.mean(pre_cor_vec) == 0:
            prob_winstay = np.nan
        else:
            prob_winstay = np.mean(idx_stay & pre_cor_vec) / np.mean(pre_cor_vec)
        return prob_winstay

    @staticmethod
    # calculating probability of LoseSwitch
    def prob_loseswitch(cor_vec, pre_cor_vec):

        idx_switch = np.array(pre_cor_vec != cor_vec)
        pre_false_vec = ~(pre_cor_vec.astype(bool)) * 1
        if np.mean(pre_false_vec) == 0:
            prob_loseswitch = np.nan
        else:
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

        # Load .mat file
        beh_data = scipy.io.loadmat(file_directory + subject + '.mat')
        num_tar_att = np.array(beh_data["NumTargetAttended"])
        att_side = np.array(beh_data["AttendedSide"])
        att_side = np.transpose(att_side)
        conf_val = np.array(beh_data["ConfidenceValue"])
        get_rew = np.array(beh_data["GetReward"])
        rew_val = np.array(beh_data["RewardVecValue"])
        att_first = np.array(beh_data["AttendedFirst"])
        sub_rt = np.array(beh_data["SubjectRT"])

        # make a dictionary for necessary variables
        tmp_beh_vars = {"att_side": att_side, "conf_val": conf_val, "get_rew": get_rew, "rew_val": rew_val,
                        "sub_rt": sub_rt, "att_first": att_first, "num_tar_att": num_tar_att}

        # append counters data together
        beh_vars.append(tmp_beh_vars)

    return beh_vars


# define the function which makes the data table of subjects
def table_maker(beh_vars):
    """

    :param beh_vars: list of the data of the subjects
    :return: a dictionary of the data
    """
    import pandas as pd
    from scipy import stats
    table_data = pd.DataFrame()
    for subject in range(len(beh_vars)):

        tmp_sub = beh_vars[subject]
        num_trial = tmp_sub["att_first"].size
        num_block = tmp_sub["att_side"].size
        att_side = tmp_sub["att_side"]
        att_side = np.repeat(att_side, num_trial/num_block)
        sub_num = np.repeat(subject, num_trial)

        # make a dictionary for necessary variables
        tmp_data = {"sub_num": sub_num, "att_side": att_side, "conf_val": tmp_sub["conf_val"].flatten(),
                    "get_rew": tmp_sub["get_rew"].flatten(), "rew_val": tmp_sub["rew_val"].flatten(),
                    "sub_rt": tmp_sub["sub_rt"].flatten(), "att_first": tmp_sub["att_first"].flatten(),
                    "num_tar_att": tmp_sub["num_tar_att"].flatten(), "z_rt": stats.zscore(tmp_sub["sub_rt"].flatten())}

        tmp_table_data = pd.DataFrame(data=tmp_data)
        table_data = table_data.append(tmp_table_data, ignore_index=True)

    return table_data


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
    # check if the inputs are legitimate
    if (idx_rew not in [0, 1, 3]) and (idx_side in [0, 1, 2]) and (idx_conf in [0, 1, 2]) and \
            (idx_att_first in [0, 1, 2]):
        er_var = 'idx_rew'
        er_exist = True
    elif (idx_rew in [0, 1, 3]) and (idx_side not in [0, 1, 2]) and (idx_conf in [0, 1, 2]) and \
            (idx_att_first in [0, 1, 2]):
        er_var = 'idx_side'
        er_exist = True
    elif (idx_rew in [0, 1, 3]) and (idx_side in [0, 1, 2]) and (idx_conf not in [0, 1, 2]) and \
            (idx_att_first in [0, 1, 2]):
        er_var = 'idx_conf'
        er_exist = True
    elif (idx_rew in [0, 1, 3]) and (idx_side in [0, 1, 2]) and (idx_conf in [0, 1, 2]) and \
            (idx_att_first not in [0, 1, 2]):
        er_var = 'idx_att_first'
        er_exist = True
    elif (idx_rew in [0, 1, 3]) and (idx_side in [0, 1, 2]) and (idx_conf in [0, 1, 2]) and \
            (idx_att_first in [0, 1, 2]):
        er_exist = False
    else:
        er_var = 'Unknown'
        er_exist = True

    if er_exist:
        raise ValueError('Invalid value for {}'.format(er_var))

    # separate the blocks we need
    if idx_side == 1 or idx_side == 2:
        num_block = int((beh_vars[0]["att_side"].shape[0]) / 2)
    else:
        num_block = int(beh_vars[0]["att_side"].shape[0])

    # initialization of matrices
    performance = np.nan * np.zeros(shape=(len(beh_vars), num_block))
    prob_stay = np.nan * np.zeros(shape=(len(beh_vars), num_block))
    prob_winstay = np.nan * np.zeros(shape=(len(beh_vars), num_block))
    prob_loseswitch = np.nan * np.zeros(shape=(len(beh_vars), num_block))
    mean_sub_rt = np.nan * np.zeros(shape=(len(beh_vars), num_block))
    sub_rt = []

    cnt_sub = 0
    for sub_beh in beh_vars:

        tmp_beh_data1 = {}

        if idx_side == 1 or idx_side == 2:

            idx_side_block = np.where(sub_beh["att_side"] == idx_side)[0]

            for key in sub_beh.keys():
                tmp_beh_data1[key] = sub_beh[key][idx_side_block, :]
        else:
            tmp_beh_data1 = sub_beh

        for block in range(num_block):
            # calculate the average of correct over reward and confidence conditions
            if (idx_rew == 1 or idx_rew == 3) and (idx_conf != 2 and idx_conf != 1) and (
                    idx_att_first != 1 and idx_att_first != 0):

                idx_sel_bool = tmp_beh_data1["rew_val"][block, :] == idx_rew

            elif (idx_rew != 1 and idx_rew != 3) and (idx_conf == 2 or idx_conf == 1) and (
                    idx_att_first != 1 and idx_att_first != 0):

                idx_sel_bool = tmp_beh_data1["conf_val"][block, :] == idx_conf

            elif (idx_rew != 1 and idx_rew != 3) and (idx_conf != 2 and idx_conf != 1) and (
                    idx_att_first == 1 or idx_att_first == 0):

                idx_sel_bool = tmp_beh_data1["att_first"][block, :] == idx_att_first

            elif (idx_rew == 1 or idx_rew == 3) and (idx_conf == 2 or idx_conf == 1) and (
                    idx_att_first != 1 and idx_att_first != 0):

                idx_sel_bool = (tmp_beh_data1["conf_val"][block, :] == idx_conf) & \
                               (tmp_beh_data1["rew_val"][block, :] == idx_rew)

            elif (idx_rew == 1 or idx_rew == 3) and (idx_conf != 2 and idx_conf != 1) and (
                    idx_att_first == 1 or idx_att_first == 0):

                idx_sel_bool = (tmp_beh_data1["rew_val"][block, :] == idx_rew) & \
                               (tmp_beh_data1["att_first"][block, :] == idx_att_first)

            elif (idx_rew != 1 and idx_rew != 3) and (idx_conf == 2 or idx_conf == 1) and (
                    idx_att_first == 1 or idx_att_first == 0):

                idx_sel_bool = (tmp_beh_data1["conf_val"][block, :] == idx_conf) & \
                               (tmp_beh_data1["att_first"][block, :] == idx_att_first)

            elif (idx_rew == 1 or idx_rew == 3) and (idx_conf == 2 or idx_conf == 1) and (
                    idx_att_first == 1 or idx_att_first == 0):

                idx_sel_bool = (tmp_beh_data1["conf_val"][block, :] == idx_conf) & \
                               (tmp_beh_data1["rew_val"][block, :] == idx_rew) & \
                               (tmp_beh_data1["att_first"][block, :] == idx_att_first)
            else:

                idx_sel_bool = np.ones((len(tmp_beh_data1["rew_val"][block, :]), 1), dtype=bool)

            # keeping only the trials with one target
            idx_sel_bool = idx_sel_bool.reshape(idx_sel_bool.shape[0], 1)
            tmp_cor_vec = (tmp_beh_data1["get_rew"][block, :])
            tmp_cor_vec = tmp_cor_vec.reshape(tmp_cor_vec.shape[0], 1)
            tmp_num_tar = (tmp_beh_data1["num_tar_att"][block, :])
            tmp_num_tar = tmp_num_tar.reshape(tmp_num_tar.shape[0], 1)
            idx_one_target = tmp_num_tar == 1
            idx_tar = (idx_one_target & idx_sel_bool)
            cor_vec = tmp_cor_vec[idx_tar]
            idx_pre = np.insert(idx_tar[:-1], 0, True)
            # since previous trial could have 2 reward I just make all 2's to be also 1 for stay and winstay
            pre_cor_vec = (np.transpose(tmp_cor_vec[idx_pre]) > 0).astype(int)
            performance[cnt_sub, block] = Behavior.performance(cor_vec)
            prob_stay[cnt_sub, block] = Behavior.prob_stay(cor_vec, pre_cor_vec)
            prob_winstay[cnt_sub, block] = Behavior.prob_winstay(cor_vec, pre_cor_vec)
            prob_loseswitch[cnt_sub, block] = Behavior.prob_loseswitch(cor_vec, pre_cor_vec)
            tmp_rt = tmp_beh_data1["sub_rt"][block, :]
            tmp_rt = tmp_rt.reshape(tmp_rt.shape[0], 1)
            tmp_rt = tmp_rt[idx_tar & tmp_cor_vec > 0]
            tmp_rt = tmp_rt[tmp_rt > 0]  # remove the ones which was no answer or negative RT (answering before target)

            if any(tmp_rt > 1):
                raise ValueError('RT could not be higher than 1sec')
            sub_rt.append(tmp_rt)
            mean_sub_rt[cnt_sub, block] = np.mean(tmp_rt)
        # add one to the counter of subjects
        cnt_sub += 1
    beh_result = {"performance": performance, "prob_stay": prob_stay, "prob_winstay": prob_winstay,
                  "prob_loseswitch": prob_loseswitch, "sub_rt": sub_rt, "mean_sub_rt": mean_sub_rt}

    return beh_result


