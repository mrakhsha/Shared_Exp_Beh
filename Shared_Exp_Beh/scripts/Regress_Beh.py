import Shared_Exp_Beh as seb
import os.path as op
import statsmodels.formula.api as smf

data_path = op.join(seb.__path__[0], 'data/')

file_directory = data_path
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

beh_vars = seb.var_extractor(file_directory, subject_list)
# making a table from all the trials of all subjects
table_data = seb.table_maker(beh_vars)
table_data = table_data[table_data["num_tar_att"] == 1]

# regression for accuracy
res1 = smf.ols(formula='get_rew ~ att_first * att_side + conf_val * rew_val ', data=table_data)
res = res1.fit()
print(res.summary())

# regression for reaction time only on correct trials
table_data_cor = table_data[table_data["get_rew"] == 1]
res1 = smf.ols(formula='z_rt ~ att_first + att_side + conf_val * rew_val ', data=table_data_cor)
res = res1.fit()
print(res.summary())