import numpy as np
import scipy.io
import Shared_Exp_Beh as seb
import os.path as op

data_path = op.join(seb.__path__[0], 'data/')


def test_beh_analysis():
    """

    :return: Test results raise error
    """

    # Test if the size of all variables of the experiment is same
    file_directory = data_path
    subject_list = ['behav_Shared_ARSubNum21']
    beh_vars = seb.var_extractor(file_directory, subject_list)
    assert beh_vars[0]["conf_val"].shape == beh_vars[0]["conf_val"].shape == beh_vars[0]["get_rew"].shape == \
           beh_vars[0]["rew_val"].shape == beh_vars[0]["sub_rt"].shape == beh_vars[0]["att_first"].shape == \
           beh_vars[0]["num_tar_att"].shape

    # Tests of stay, winstay, and loseswitch
    cor_vec = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert seb.Behavior.performance(cor_vec) == float(0)

    cor_vec = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert seb.Behavior.performance(cor_vec) == float(100)

    cor_vec = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    assert seb.Behavior.performance(cor_vec) == float(50)

    pre_cor_vec = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    cor_vec = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert seb.Behavior.prob_stay(cor_vec, pre_cor_vec) == float(1)

    pre_cor_vec = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    cor_vec = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert seb.Behavior.prob_stay(cor_vec, pre_cor_vec) == float(1)

    pre_cor_vec = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    cor_vec = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert seb.Behavior.prob_stay(cor_vec, pre_cor_vec) == float(0)

    pre_cor_vec = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    cor_vec = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert seb.Behavior.prob_winstay(cor_vec, pre_cor_vec) == float(0)
    assert np.isnan(seb.Behavior.prob_loseswitch(cor_vec, pre_cor_vec))

    pre_cor_vec = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    cor_vec = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert seb.Behavior.prob_winstay(cor_vec, pre_cor_vec) == float(1)
    assert np.isnan(seb.Behavior.prob_loseswitch(cor_vec, pre_cor_vec))

    pre_cor_vec = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    cor_vec = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert np.isnan(seb.Behavior.prob_winstay(cor_vec, pre_cor_vec))
    assert seb.Behavior.prob_loseswitch(cor_vec, pre_cor_vec) == float(0)

    pre_cor_vec = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    cor_vec = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert np.isnan(seb.Behavior.prob_winstay(cor_vec, pre_cor_vec))
    assert seb.Behavior.prob_loseswitch(cor_vec, pre_cor_vec) == float(1)

    # smoke tests for beh_analysis
    beh0002 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=0, idx_side=0, idx_att_first=2)
    beh0012 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=0, idx_side=1, idx_att_first=2)
    beh0022 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=0, idx_side=2, idx_att_first=2)
    beh0001 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=0, idx_side=0, idx_att_first=1)
    beh0000 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=0, idx_side=0, idx_att_first=0)
    beh3002 = seb.beh_analysis(beh_vars, idx_rew=3, idx_conf=0, idx_side=0, idx_att_first=2)
    beh1002 = seb.beh_analysis(beh_vars, idx_rew=1, idx_conf=0, idx_side=0, idx_att_first=2)
    beh0202 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=2, idx_side=0, idx_att_first=2)
    beh0102 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=1, idx_side=0, idx_att_first=2)

    assert beh0002["performance"].shape == beh0002["prob_stay"].shape == beh0002["prob_winstay"].shape == \
           beh0002["prob_loseswitch"].shape == beh0002["mean_sub_rt"].shape
    assert beh0012["performance"].shape == beh0012["prob_stay"].shape == beh0012["prob_winstay"].shape == \
           beh0012["prob_loseswitch"].shape == beh0012["mean_sub_rt"].shape
    assert beh0022["performance"].shape == beh0022["prob_stay"].shape == beh0022["prob_winstay"].shape == \
           beh0022["prob_loseswitch"].shape == beh0022["mean_sub_rt"].shape
    assert beh0001["performance"].shape == beh0001["prob_stay"].shape == beh0001["prob_winstay"].shape == \
           beh0001["prob_loseswitch"].shape == beh0001["mean_sub_rt"].shape
    assert beh0000["performance"].shape == beh0000["prob_stay"].shape == beh0000["prob_winstay"].shape == \
           beh0000["prob_loseswitch"].shape == beh0000["mean_sub_rt"].shape
    assert beh3002["performance"].shape == beh3002["prob_stay"].shape == beh3002["prob_winstay"].shape == \
           beh3002["prob_loseswitch"].shape == beh3002["mean_sub_rt"].shape
    assert beh1002["performance"].shape == beh1002["prob_stay"].shape == beh1002["prob_winstay"].shape == \
           beh1002["prob_loseswitch"].shape == beh1002["mean_sub_rt"].shape
    assert beh0202["performance"].shape == beh0202["prob_stay"].shape == beh0202["prob_winstay"].shape == \
           beh0202["prob_loseswitch"].shape == beh0202["mean_sub_rt"].shape
    assert beh0102["performance"].shape == beh0102["prob_stay"].shape == beh0102["prob_winstay"].shape == \
           beh0102["prob_loseswitch"].shape == beh0102["mean_sub_rt"].shape

