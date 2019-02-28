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

    # when all the trials are correct LoseSwitch should be nan
    # when all the trials are wrong WinStay should be nan
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
    for id_rew in [0, 1, 3]:
        for id_conf in [0, 1, 2]:
            for id_side in [0, 1, 2]:
                for id_att_first in [0, 1, 2]:
                    beh = seb.beh_analysis(beh_vars, idx_rew=id_rew, idx_conf=id_conf, idx_side=id_side, idx_att_first=id_att_first)
                    assert beh["performance"].shape == beh["prob_stay"].shape == beh[
                        "prob_winstay"].shape == beh["prob_loseswitch"].shape == beh["mean_sub_rt"].shape

    # raise error for not permissible values
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

