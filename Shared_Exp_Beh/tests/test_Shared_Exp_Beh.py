import numpy as np
import scipy.io
import Shared_Exp_Beh as seb
import os.path as op
import pytest

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
    beh0000 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=0, idx_side=0, idx_att_first=0)
    beh0001 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=0, idx_side=0, idx_att_first=1)
    beh0002 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=0, idx_side=0, idx_att_first=2)
    beh0010 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=0, idx_side=1, idx_att_first=0)
    beh0011 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=0, idx_side=1, idx_att_first=1)
    beh0012 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=0, idx_side=1, idx_att_first=2)
    beh0020 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=0, idx_side=2, idx_att_first=0)
    beh0021 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=0, idx_side=2, idx_att_first=1)
    beh0022 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=0, idx_side=2, idx_att_first=2)

    beh0100 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=1, idx_side=0, idx_att_first=0)
    beh0101 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=1, idx_side=0, idx_att_first=1)
    beh0102 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=1, idx_side=0, idx_att_first=2)
    beh0110 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=1, idx_side=1, idx_att_first=0)
    beh0111 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=1, idx_side=1, idx_att_first=1)
    beh0112 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=1, idx_side=1, idx_att_first=2)
    beh0120 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=1, idx_side=2, idx_att_first=0)
    beh0121 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=1, idx_side=2, idx_att_first=1)
    beh0122 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=1, idx_side=2, idx_att_first=2)

    beh0200 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=2, idx_side=0, idx_att_first=0)
    beh0201 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=2, idx_side=0, idx_att_first=1)
    beh0202 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=2, idx_side=0, idx_att_first=2)
    beh0210 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=2, idx_side=1, idx_att_first=0)
    beh0211 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=2, idx_side=1, idx_att_first=1)
    beh0212 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=2, idx_side=1, idx_att_first=2)
    beh0220 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=2, idx_side=2, idx_att_first=0)
    beh0221 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=2, idx_side=2, idx_att_first=1)
    beh0222 = seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=2, idx_side=2, idx_att_first=2)

    beh1000 = seb.beh_analysis(beh_vars, idx_rew=1, idx_conf=0, idx_side=0, idx_att_first=0)
    beh1001 = seb.beh_analysis(beh_vars, idx_rew=1, idx_conf=0, idx_side=0, idx_att_first=1)
    beh1002 = seb.beh_analysis(beh_vars, idx_rew=1, idx_conf=0, idx_side=0, idx_att_first=2)
    beh1010 = seb.beh_analysis(beh_vars, idx_rew=1, idx_conf=0, idx_side=1, idx_att_first=0)
    beh1011 = seb.beh_analysis(beh_vars, idx_rew=1, idx_conf=0, idx_side=1, idx_att_first=1)
    beh1012 = seb.beh_analysis(beh_vars, idx_rew=1, idx_conf=0, idx_side=1, idx_att_first=2)
    beh1020 = seb.beh_analysis(beh_vars, idx_rew=1, idx_conf=0, idx_side=2, idx_att_first=0)
    beh1021 = seb.beh_analysis(beh_vars, idx_rew=1, idx_conf=0, idx_side=2, idx_att_first=1)
    beh1022 = seb.beh_analysis(beh_vars, idx_rew=1, idx_conf=0, idx_side=2, idx_att_first=2)

    beh1100 = seb.beh_analysis(beh_vars, idx_rew=1, idx_conf=1, idx_side=0, idx_att_first=0)
    beh1101 = seb.beh_analysis(beh_vars, idx_rew=1, idx_conf=1, idx_side=0, idx_att_first=1)
    beh1102 = seb.beh_analysis(beh_vars, idx_rew=1, idx_conf=1, idx_side=0, idx_att_first=2)
    beh1110 = seb.beh_analysis(beh_vars, idx_rew=1, idx_conf=1, idx_side=1, idx_att_first=0)
    beh1111 = seb.beh_analysis(beh_vars, idx_rew=1, idx_conf=1, idx_side=1, idx_att_first=1)
    beh1112 = seb.beh_analysis(beh_vars, idx_rew=1, idx_conf=1, idx_side=1, idx_att_first=2)
    beh1120 = seb.beh_analysis(beh_vars, idx_rew=1, idx_conf=1, idx_side=2, idx_att_first=0)
    beh1121 = seb.beh_analysis(beh_vars, idx_rew=1, idx_conf=1, idx_side=2, idx_att_first=1)
    beh1122 = seb.beh_analysis(beh_vars, idx_rew=1, idx_conf=1, idx_side=2, idx_att_first=2)

    beh1200 = seb.beh_analysis(beh_vars, idx_rew=1, idx_conf=2, idx_side=0, idx_att_first=0)
    beh1201 = seb.beh_analysis(beh_vars, idx_rew=1, idx_conf=2, idx_side=0, idx_att_first=1)
    beh1202 = seb.beh_analysis(beh_vars, idx_rew=1, idx_conf=2, idx_side=0, idx_att_first=2)
    beh1210 = seb.beh_analysis(beh_vars, idx_rew=1, idx_conf=2, idx_side=1, idx_att_first=0)
    beh1211 = seb.beh_analysis(beh_vars, idx_rew=1, idx_conf=2, idx_side=1, idx_att_first=1)
    beh1212 = seb.beh_analysis(beh_vars, idx_rew=1, idx_conf=2, idx_side=1, idx_att_first=2)
    beh1220 = seb.beh_analysis(beh_vars, idx_rew=1, idx_conf=2, idx_side=2, idx_att_first=0)
    beh1221 = seb.beh_analysis(beh_vars, idx_rew=1, idx_conf=2, idx_side=2, idx_att_first=1)
    beh1222 = seb.beh_analysis(beh_vars, idx_rew=1, idx_conf=2, idx_side=2, idx_att_first=2)

    beh3000 = seb.beh_analysis(beh_vars, idx_rew=3, idx_conf=0, idx_side=0, idx_att_first=0)
    beh3001 = seb.beh_analysis(beh_vars, idx_rew=3, idx_conf=0, idx_side=0, idx_att_first=1)
    beh3002 = seb.beh_analysis(beh_vars, idx_rew=3, idx_conf=0, idx_side=0, idx_att_first=2)
    beh3010 = seb.beh_analysis(beh_vars, idx_rew=3, idx_conf=0, idx_side=1, idx_att_first=0)
    beh3011 = seb.beh_analysis(beh_vars, idx_rew=3, idx_conf=0, idx_side=1, idx_att_first=1)
    beh3012 = seb.beh_analysis(beh_vars, idx_rew=3, idx_conf=0, idx_side=1, idx_att_first=2)
    beh3020 = seb.beh_analysis(beh_vars, idx_rew=3, idx_conf=0, idx_side=2, idx_att_first=0)
    beh3021 = seb.beh_analysis(beh_vars, idx_rew=3, idx_conf=0, idx_side=2, idx_att_first=1)
    beh3022 = seb.beh_analysis(beh_vars, idx_rew=3, idx_conf=0, idx_side=2, idx_att_first=2)

    beh3100 = seb.beh_analysis(beh_vars, idx_rew=3, idx_conf=1, idx_side=0, idx_att_first=0)
    beh3101 = seb.beh_analysis(beh_vars, idx_rew=3, idx_conf=1, idx_side=0, idx_att_first=1)
    beh3102 = seb.beh_analysis(beh_vars, idx_rew=3, idx_conf=1, idx_side=0, idx_att_first=2)
    beh3110 = seb.beh_analysis(beh_vars, idx_rew=3, idx_conf=1, idx_side=1, idx_att_first=0)
    beh3111 = seb.beh_analysis(beh_vars, idx_rew=3, idx_conf=1, idx_side=1, idx_att_first=1)
    beh3112 = seb.beh_analysis(beh_vars, idx_rew=3, idx_conf=1, idx_side=1, idx_att_first=2)
    beh3120 = seb.beh_analysis(beh_vars, idx_rew=3, idx_conf=1, idx_side=2, idx_att_first=0)
    beh3121 = seb.beh_analysis(beh_vars, idx_rew=3, idx_conf=1, idx_side=2, idx_att_first=1)
    beh3122 = seb.beh_analysis(beh_vars, idx_rew=3, idx_conf=1, idx_side=2, idx_att_first=2)

    beh3200 = seb.beh_analysis(beh_vars, idx_rew=3, idx_conf=2, idx_side=0, idx_att_first=0)
    beh3201 = seb.beh_analysis(beh_vars, idx_rew=3, idx_conf=2, idx_side=0, idx_att_first=1)
    beh3202 = seb.beh_analysis(beh_vars, idx_rew=3, idx_conf=2, idx_side=0, idx_att_first=2)
    beh3210 = seb.beh_analysis(beh_vars, idx_rew=3, idx_conf=2, idx_side=1, idx_att_first=0)
    beh3211 = seb.beh_analysis(beh_vars, idx_rew=3, idx_conf=2, idx_side=1, idx_att_first=1)
    beh3212 = seb.beh_analysis(beh_vars, idx_rew=3, idx_conf=2, idx_side=1, idx_att_first=2)
    beh3220 = seb.beh_analysis(beh_vars, idx_rew=3, idx_conf=2, idx_side=2, idx_att_first=0)
    beh3221 = seb.beh_analysis(beh_vars, idx_rew=3, idx_conf=2, idx_side=2, idx_att_first=1)
    beh3222 = seb.beh_analysis(beh_vars, idx_rew=3, idx_conf=2, idx_side=2, idx_att_first=2)

    assert beh0000["performance"].shape == beh0000["prob_stay"].shape == beh0000["prob_winstay"].shape == \
           beh0000["prob_loseswitch"].shape == beh0000["mean_sub_rt"].shape
    assert beh0001["performance"].shape == beh0001["prob_stay"].shape == beh0001["prob_winstay"].shape == \
           beh0001["prob_loseswitch"].shape == beh0001["mean_sub_rt"].shape
    assert beh0002["performance"].shape == beh0002["prob_stay"].shape == beh0002["prob_winstay"].shape == \
           beh0002["prob_loseswitch"].shape == beh0002["mean_sub_rt"].shape

    assert beh0010["performance"].shape == beh0010["prob_stay"].shape == beh0010["prob_winstay"].shape == \
           beh0010["prob_loseswitch"].shape == beh0010["mean_sub_rt"].shape
    assert beh0011["performance"].shape == beh0011["prob_stay"].shape == beh0011["prob_winstay"].shape == \
           beh0011["prob_loseswitch"].shape == beh0011["mean_sub_rt"].shape
    assert beh0012["performance"].shape == beh0012["prob_stay"].shape == beh0012["prob_winstay"].shape == \
           beh0012["prob_loseswitch"].shape == beh0012["mean_sub_rt"].shape

    assert beh0020["performance"].shape == beh0020["prob_stay"].shape == beh0020["prob_winstay"].shape == \
           beh0020["prob_loseswitch"].shape == beh0020["mean_sub_rt"].shape
    assert beh0021["performance"].shape == beh0021["prob_stay"].shape == beh0021["prob_winstay"].shape == \
           beh0021["prob_loseswitch"].shape == beh0021["mean_sub_rt"].shape
    assert beh0022["performance"].shape == beh0022["prob_stay"].shape == beh0022["prob_winstay"].shape == \
           beh0022["prob_loseswitch"].shape == beh0022["mean_sub_rt"].shape

    assert beh0100["performance"].shape == beh0100["prob_stay"].shape == beh0100["prob_winstay"].shape == \
           beh0100["prob_loseswitch"].shape == beh0100["mean_sub_rt"].shape
    assert beh0101["performance"].shape == beh0101["prob_stay"].shape == beh0101["prob_winstay"].shape == \
           beh0101["prob_loseswitch"].shape == beh0101["mean_sub_rt"].shape
    assert beh0102["performance"].shape == beh0102["prob_stay"].shape == beh0102["prob_winstay"].shape == \
           beh0102["prob_loseswitch"].shape == beh0102["mean_sub_rt"].shape

    assert beh0110["performance"].shape == beh0110["prob_stay"].shape == beh0110["prob_winstay"].shape == \
           beh0110["prob_loseswitch"].shape == beh0110["mean_sub_rt"].shape
    assert beh0111["performance"].shape == beh0111["prob_stay"].shape == beh0111["prob_winstay"].shape == \
           beh0111["prob_loseswitch"].shape == beh0111["mean_sub_rt"].shape
    assert beh0112["performance"].shape == beh0112["prob_stay"].shape == beh0112["prob_winstay"].shape == \
           beh0112["prob_loseswitch"].shape == beh0112["mean_sub_rt"].shape

    assert beh0120["performance"].shape == beh0120["prob_stay"].shape == beh0120["prob_winstay"].shape == \
           beh0120["prob_loseswitch"].shape == beh0120["mean_sub_rt"].shape
    assert beh0121["performance"].shape == beh0121["prob_stay"].shape == beh0121["prob_winstay"].shape == \
           beh0121["prob_loseswitch"].shape == beh0121["mean_sub_rt"].shape
    assert beh0122["performance"].shape == beh0122["prob_stay"].shape == beh0122["prob_winstay"].shape == \
           beh0122["prob_loseswitch"].shape == beh0122["mean_sub_rt"].shape

    assert beh0200["performance"].shape == beh0200["prob_stay"].shape == beh0200["prob_winstay"].shape == \
           beh0200["prob_loseswitch"].shape == beh0200["mean_sub_rt"].shape
    assert beh0201["performance"].shape == beh0201["prob_stay"].shape == beh0201["prob_winstay"].shape == \
           beh0201["prob_loseswitch"].shape == beh0201["mean_sub_rt"].shape
    assert beh0202["performance"].shape == beh0202["prob_stay"].shape == beh0202["prob_winstay"].shape == \
           beh0202["prob_loseswitch"].shape == beh0202["mean_sub_rt"].shape

    assert beh0210["performance"].shape == beh0210["prob_stay"].shape == beh0210["prob_winstay"].shape == \
           beh0210["prob_loseswitch"].shape == beh0210["mean_sub_rt"].shape
    assert beh0211["performance"].shape == beh0211["prob_stay"].shape == beh0211["prob_winstay"].shape == \
           beh0211["prob_loseswitch"].shape == beh0211["mean_sub_rt"].shape
    assert beh0212["performance"].shape == beh0212["prob_stay"].shape == beh0212["prob_winstay"].shape == \
           beh0212["prob_loseswitch"].shape == beh0212["mean_sub_rt"].shape

    assert beh0220["performance"].shape == beh0220["prob_stay"].shape == beh0220["prob_winstay"].shape == \
           beh0220["prob_loseswitch"].shape == beh0220["mean_sub_rt"].shape
    assert beh0221["performance"].shape == beh0221["prob_stay"].shape == beh0221["prob_winstay"].shape == \
           beh0221["prob_loseswitch"].shape == beh0221["mean_sub_rt"].shape
    assert beh0222["performance"].shape == beh0222["prob_stay"].shape == beh0222["prob_winstay"].shape == \
           beh0222["prob_loseswitch"].shape == beh0222["mean_sub_rt"].shape

    with pytest.raises(ValueError):
        seb.beh_analysis(beh_vars, idx_rew=4, idx_conf=0, idx_side=0, idx_att_first=2)
    with pytest.raises(ValueError):
        seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=4, idx_side=0, idx_att_first=2)
    with pytest.raises(ValueError):
        seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=0, idx_side=4, idx_att_first=2)
    with pytest.raises(ValueError):
        seb.beh_analysis(beh_vars, idx_rew=0, idx_conf=0, idx_side=1, idx_att_first=4)
    with pytest.raises(ValueError):
        seb.beh_analysis(beh_vars, idx_rew=4, idx_conf=4, idx_side=4, idx_att_first=4)

