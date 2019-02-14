"""
This code is written for behavioral analysis of the EEG shared experiment
Written by Mohsen Rakhshan 
V2 . The first version (Dec. 20, 2018)

To do list: I should calculate WinStay LoseSwitch and the others on the real indices and not in the IDSelective
"""
# import necessary libraries and packages
import scipy.io
import numpy as np
import pdb
import dill

# Directory of the behavioral data
FileDirectory = '/Users/mohsen/Dropbox (CCNL-Dartmouth)/Shared Experiments REWARD/BehData/';
# list of subjects
SubjectsList = ['behav_Shared_ARSubNum21', 'behav_Shared_ESSubNum24', 'behav_Shared_HASubNum20',
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


# SubjectsList  = ['Shared_GFSubNum15']
# SubjectsList  = ['Shared_GFSubNum15','Shared_KKSubNum14','Shared_NH2SubNum13']
# %%
###############################################################################
############################# Functions #######################################
###############################################################################

# define the function which extracts the behavioral variables we need
def VariableExtractor(FileDirectory, SubjectsList):
    BehVariables = []
    # main loop for loading the data
    for Counter_Subjects in range(len(SubjectsList)):
        BehData = scipy.io.loadmat(FileDirectory + SubjectsList[Counter_Subjects] + '.mat')
        # pdb.set_trace()
        NumTargetAttended = np.array(BehData["NumTargetAttended"])
        AttendedSide = np.array(BehData["AttendedSide"])
        AttendedSide = np.transpose(AttendedSide)
        ConfidenceValue = np.array(BehData["ConfidenceValue"])
        GetReward = np.array(BehData["GetReward"])
        RewardVecValue = np.array(BehData["RewardVecValue"])
        AttendedFirst = np.array(BehData["AttendedFirst"])
        SubjectRT = np.array(BehData["SubjectRT"])

        # Extracting the ones with only one target
        IDOneTarget = NumTargetAttended == 1
        # pdb.set_trace()
        tmpConfidenceValue = ConfidenceValue[IDOneTarget]
        tmpGetReward = GetReward[IDOneTarget]
        tmpRewardVecValue = RewardVecValue[IDOneTarget]
        tmpSubjectRT = SubjectRT[IDOneTarget]
        tmpAttendedFirst = AttendedFirst[IDOneTarget]
        # pdb.set_trace()
        # reshape to the block form
        ConfidenceValue = tmpConfidenceValue.reshape(AttendedSide.shape[0],
                                                     int(tmpConfidenceValue.shape[0] / AttendedSide.shape[0]))
        GetReward = tmpGetReward.reshape(AttendedSide.shape[0], int(tmpGetReward.shape[0] / AttendedSide.shape[0]))
        RewardVecValue = tmpRewardVecValue.reshape(AttendedSide.shape[0],
                                                   int(tmpRewardVecValue.shape[0] / AttendedSide.shape[0]))
        SubjectRT = tmpSubjectRT.reshape(AttendedSide.shape[0], int(tmpSubjectRT.shape[0] / AttendedSide.shape[0]))
        AttendedFirst = tmpAttendedFirst.reshape(AttendedSide.shape[0],
                                                 int(tmpAttendedFirst.shape[0] / AttendedSide.shape[0]))
        # make a dictionary for necessary variables
        tmpBehVariables = {"AttendedSide": AttendedSide, "ConfidenceValue": ConfidenceValue, "GetReward": GetReward,
                           "RewardVecValue": RewardVecValue, "SubjectRT": SubjectRT, "AttendedFirst": AttendedFirst}
        # append counters data together
        BehVariables.append(tmpBehVariables)

    return BehVariables


###############################################################################

# define a class of functions for behavior analysis
class Behavior:
    ...

    @staticmethod
    # calculating correct
    def Performance(CorrectVec):
        MeanPerformance = (np.nanmean(CorrectVec)) * 100
        return MeanPerformance

    # calculating probability of stay
    def PStay(CorrectVec):
        PreCorrectVec = np.insert(CorrectVec[:-1], 0, 0)
        IDStay = np.array(PreCorrectVec == CorrectVec)
        PStay = np.mean(IDStay)
        return PStay

    # calculating probability of WinStay
    def PWinStay(CorrectVec):
        PreCorrectVec = np.insert(CorrectVec[:-1], 0, 0)
        IDStay = np.array(PreCorrectVec == CorrectVec)
        PWinStay = np.mean(IDStay & PreCorrectVec) / np.mean(PreCorrectVec)
        return PWinStay

    # calculating probability of LoseSwitch
    def PLoseSwitch(CorrectVec):
        PreCorrectVec = np.insert(CorrectVec[:-1], 0, 0)
        IDSwitch = np.array(PreCorrectVec != CorrectVec)
        PreFalseVec = ~(PreCorrectVec.astype(bool)) * 1
        PLoseSwitch = np.mean(IDSwitch & PreFalseVec) / np.mean(PreFalseVec)
        return PLoseSwitch


###############################################################################

def BehAnalysis(BehVariables, IndexRew, IndexConf, IndexSide, IndexAttendedFirst):
    if IndexSide == 1 or IndexSide == 2:

        NumBlock = int((BehVariables[0]["AttendedSide"].shape[0]) / 2)
    else:
        NumBlock = int(BehVariables[0]["AttendedSide"].shape[0])

    # pdb.set_trace()
    # initialization
    Performance = np.nan * np.zeros(shape=(len(BehVariables), NumBlock))
    PStay = np.nan * np.zeros(shape=(len(BehVariables), NumBlock))
    PWinStay = np.nan * np.zeros(shape=(len(BehVariables), NumBlock))
    PLoseSwitch = np.nan * np.zeros(shape=(len(BehVariables), NumBlock))
    SubjectRT = []
    MeanSubjectRT = np.nan * np.zeros(shape=(len(BehVariables), NumBlock))

    for Counter_Subjects in range(len(BehVariables)):

        tmpBehData = BehVariables[Counter_Subjects]
        tmpBehData1 = {}

        if IndexSide == 1 or IndexSide == 2:

            IDSide = np.where(BehVariables[0]["AttendedSide"] == IndexSide)[0]

            for Counter_DictKeys in tmpBehData.keys():
                tmpBehData1[Counter_DictKeys] = tmpBehData[Counter_DictKeys][IDSide, :]
        else:
            tmpBehData1 = tmpBehData

        for Counter_Block in range(NumBlock):

            # pdb.set_trace()
            # calculate the average of correct over reward and confidence conditions
            if (IndexRew == 1 or IndexRew == 3) and (IndexConf != 2 and IndexConf != 1) and (
                    IndexAttendedFirst != 1 and IndexAttendedFirst != 0):
                # pdb.set_trace()
                IDSelective = np.where(tmpBehData1["RewardVecValue"][Counter_Block, :] == IndexRew)

            elif (IndexRew != 1 and IndexRew != 3) and (IndexConf == 2 or IndexConf == 1) and (
                    IndexAttendedFirst != 1 and IndexAttendedFirst != 0):
                #                pdb.set_trace()
                # pdb.set_trace()
                IDSelective = np.where(tmpBehData1["ConfidenceValue"][Counter_Block, :] == IndexConf)

            elif (IndexRew != 1 and IndexRew != 3) and (IndexConf != 2 and IndexConf != 1) and (
                    IndexAttendedFirst == 1 or IndexAttendedFirst == 0):
                # pdb.set_trace()
                IDSelective = np.where(tmpBehData1["AttendedFirst"][Counter_Block, :] == IndexAttendedFirst)

            elif (IndexRew == 1 or IndexRew == 3) and (IndexConf == 2 or IndexConf == 1) and (
                    IndexAttendedFirst != 1 and IndexAttendedFirst != 0):
                # pdb.set_trace()
                IDSelective = np.intersect1d(np.where(tmpBehData1["ConfidenceValue"][Counter_Block, :] == IndexConf),
                                             np.where(tmpBehData1["RewardVecValue"][Counter_Block, :] == IndexRew))

            elif (IndexRew == 1 or IndexRew == 3) and (IndexConf != 2 and IndexConf != 1) and (
                    IndexAttendedFirst == 1 or IndexAttendedFirst == 0):
                # pdb.set_trace()
                IDSelective = np.intersect1d(np.where(tmpBehData1["RewardVecValue"][Counter_Block, :] == IndexRew),
                                             np.where(
                                                 tmpBehData1["AttendedFirst"][Counter_Block, :] == IndexAttendedFirst))

            elif (IndexRew != 1 and IndexRew != 3) and (IndexConf == 2 or IndexConf == 1) and (
                    IndexAttendedFirst == 1 or IndexAttendedFirst == 0):
                # pdb.set_trace()
                IDSelective = np.intersect1d(np.where(tmpBehData1["ConfidenceValue"][Counter_Block, :] == IndexConf),
                                             np.where(
                                                 tmpBehData1["AttendedFirst"][Counter_Block, :] == IndexAttendedFirst))

            elif (IndexRew == 1 or IndexRew == 3) and (IndexConf == 2 or IndexConf == 1) and (
                    IndexAttendedFirst == 1 or IndexAttendedFirst == 0):
                # pdb.set_trace()
                IDSelective = reduce(np.intersect1d, (
                    np.where(tmpBehData1["ConfidenceValue"][Counter_Block, :] == IndexConf),
                    np.where(tmpBehData1["RewardVecValue"][Counter_Block, :] == IndexRew),
                    np.where(tmpBehData1["AttendedFirst"][Counter_Block, :] == IndexAttendedFirst)))

            else:

                IDSelective = range(len(tmpBehData1["RewardVecValue"][Counter_Block, :]))

            # pdb.set_trace()
            CorrectVec = tmpBehData1["GetReward"][Counter_Block, IDSelective]
            Performance[Counter_Subjects, Counter_Block] = Behavior.Performance(CorrectVec)
            PStay[Counter_Subjects, Counter_Block] = Behavior.PStay(CorrectVec)
            PWinStay[Counter_Subjects, Counter_Block] = Behavior.PWinStay(CorrectVec)
            PLoseSwitch[Counter_Subjects, Counter_Block] = Behavior.PLoseSwitch(CorrectVec)
            # pdb.set_trace()
            tmpRT = tmpBehData1["SubjectRT"][Counter_Block, IDSelective][CorrectVec.astype(bool)]
            tmpRT = tmpRT[tmpRT > 0]  # remove the ones which was no answer or negative
            SubjectRT.append(tmpRT)
            MeanSubjectRT[Counter_Subjects, Counter_Block] = np.mean(tmpRT)

            # if IndexRew == 1 or IndexRew == 3:

        BehResult = {"Performance": Performance, "PStay": PStay, "PWinStay": PWinStay, "PLoseSwitch": PLoseSwitch,
                     "SubjectRT": SubjectRT, "MeanSubjectRT": MeanSubjectRT}

    return BehResult


############################################################################### 

# %%
###############################################################################
############################# Main Script #####################################
###############################################################################


# Extracting the behavioral variables
BehVariables = VariableExtractor(FileDirectory, SubjectsList)

############################# Instruction #####################################

# IndexSide = 0 means all the trials
# IndexSide = 1 means the trials of side 1
# IndexSide = 2 means the trials of side 2

# IndexRew = 0 means all the trials
# IndexRew = 3 means the trials with reward 3
# IndexRew = 1 means the trials with reward 1

# IndexConf = 0 means all the trials
# IndexConf = 2 means the trials with confidence 2
# IndexConf = 1 means the trials with confidence 1

# IndexAttendedFirst = 2
# IndexAttendedFirst = 1
# IndexAttendedFirst = 0
# %%
######################## Strategy over all trials #############################
IndexRew = 0
IndexConf = 0
IndexSide = 0
IndexAttendedFirst = 2;

BehResultAllSideAllRewAllConfAllAttend = BehAnalysis(BehVariables, IndexRew, IndexConf, IndexSide, IndexAttendedFirst)
# %%
# pdb.set_trace()

######################## Strategy over all trials #############################
IndexRew = 0
IndexConf = 0
IndexSide = 1
IndexAttendedFirst = 2;

BehResult1SideAllRewAllConfAllAttend = BehAnalysis(BehVariables, IndexRew, IndexConf, IndexSide, IndexAttendedFirst)

#
IndexRew = 0
IndexConf = 0
IndexSide = 2
IndexAttendedFirst = 2;

BehResult2SideAllRewAllConfAllAttend = BehAnalysis(BehVariables, IndexRew, IndexConf, IndexSide, IndexAttendedFirst)

Perf1 = BehResult1SideAllRewAllConfAllAttend["Performance"]
Perf2 = BehResult2SideAllRewAllConfAllAttend["Performance"]

MeanPerf1 = np.mean(Perf1, 1)
MeanPerf2 = np.mean(Perf2, 1)
TtestSideMean12 = stats.ttest_rel(np.delete(MeanPerf1, 18), np.delete(MeanPerf2, 18))

from scipy import stats

Perf1 = Perf1.reshape(Perf1.shape[0] * Perf1.shape[1], 1)
Perf2 = Perf2.reshape(Perf2.shape[0] * Perf2.shape[1], 1)
TtestSide12 = stats.ttest_rel(Perf1, Perf2)
RanksumtestSide12 = stats.ranksums(Perf1, Perf2)

# %%
############## Strategy over all trials attend first ##########################
IndexRew = 0
IndexConf = 0
IndexSide = 0
IndexAttendedFirst = 1;

BehResultAllSideAllRewAllConfAttendFirst = BehAnalysis(BehVariables, IndexRew, IndexConf, IndexSide, IndexAttendedFirst)

# %%
############## Strategy over all trials attend second #########################
IndexRew = 0
IndexConf = 0
IndexSide = 0
IndexAttendedFirst = 0;

BehResultAllSideAllRewAllConfAttendSec = BehAnalysis(BehVariables, IndexRew, IndexConf, IndexSide, IndexAttendedFirst)

# %%
############# Strategy over all trials highreward #############################
IndexRew = 3
IndexConf = 0
IndexSide = 0
IndexAttendedFirst = 2;

BehResultAllSideHighRewAllConfAllAttend = BehAnalysis(BehVariables, IndexRew, IndexConf, IndexSide, IndexAttendedFirst)
# %%
############## Strategy over all trials lowreward #############################
IndexRew = 1
IndexConf = 0
IndexSide = 0
IndexAttendedFirst = 2;

BehResultAllSideLowRewAllConfAllAttend = BehAnalysis(BehVariables, IndexRew, IndexConf, IndexSide, IndexAttendedFirst)
# %%
############### Strategy over all trials highconfidence #######################
IndexRew = 0
IndexConf = 2
IndexSide = 0
IndexAttendedFirst = 2;

BehResultAllSideAllRewHighConfAllAttend = BehAnalysis(BehVariables, IndexRew, IndexConf, IndexSide, IndexAttendedFirst)
# %%
############## Strategy over all trials lowconfidence #########################
IndexRew = 0
IndexConf = 1
IndexSide = 0
IndexAttendedFirst = 2;

BehResultAllSideAllRewLowConfAllAttend = BehAnalysis(BehVariables, IndexRew, IndexConf, IndexSide, IndexAttendedFirst)

################################# Saving the variables ########################
# %%
# pip install dill --user
filename = 'Behavior20181220.pkl'

dill.dump_session(filename)

## and to load the session again:
# import dill
# filename = 'Behavior.pkl'
# dill.load_session(filename)
