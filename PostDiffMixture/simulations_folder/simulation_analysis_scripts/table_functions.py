import matplotlib
matplotlib.use('Agg')
import pickle
import os
import ipdb
import statsmodels.stats.power as smp
import pandas as pd 
import matplotlib.pyplot as plt 
import sys
sys.path.insert(0, '../../le_experiments/')
# print(data)
import numpy as np
import os
from scipy import stats
from matplotlib.pyplot import figure
import glob
import numpy as np
import read_config
from output_format import H_ALGO_ACTION_FAILURE, H_ALGO_ACTION_SUCCESS, H_ALGO_ACTION, H_ALGO_OBSERVED_REWARD
from output_format import H_ALGO_ESTIMATED_MU, H_ALGO_ESTIMATED_V, H_ALGO_ESTIMATED_ALPHA, H_ALGO_ESTIMATED_BETA
from output_format import H_ALGO_PROB_BEST_ACTION, H_ALGO_NUM_TRIALS
import beta_bernoulli
#import thompson_policy
from pathlib import Path

EPSILON_PROB = .000001

DESIRED_POWER = 0.8
DESIRED_ALPHA = 0.05

TABLE_3_KEY_UR = "UR-c={}n={}num_steps={}"
TABLE_3_KEY_TS = "TS-c={}n={}num_steps={}"
TABLE_3_KEY_TSPPD = "TSPPD-c={}n={}num_steps={}"
TABLE_3_KEY_ETS = "ETS-c={}n={}num_steps={}"
TABLE_3_KEY_PPDG = "PPDG-c={}n={}num_steps={}"

SMALL_SIZE = 10
MEDIUM_SIZE = 10
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=8.5)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

OUTCOME_A = "TableA-Proportions+T1vsDiff" #For Table A
OUTCOME_B = "TableB-T1vsDiff" #For Table A
OUTCOME_C = "TableC-T1vsImba"
OUTCOME_D = "TableD-Proportions+T1vsImba"


def bin_imba_apply_outcome(df = None, upper = 0.5, lower = 0.0, outcome = "Proportions", include_stderr = False):
    """
    percentage
    """
    num_sims = len(df)
    df["imba"] = np.abs(df["sample_size_1"] / (df["sample_size_1"] + df["sample_size_2"]) - 0.5)
    df["wald_reject"] = df["wald_pval"] < 0.05
    bin_curr = df[(lower <= df["imba"]) & (df["imba"] < upper)]
    t1_total = np.round(np.sum(df["wald_reject"])/num_sims, 3)

    if outcome == OUTCOME_D:
        prop = len(bin_curr)/num_sims
        t1_err = np.sum(bin_curr["wald_reject"]) / num_sims
#        t1_err = np.round(t1_err, 3)

        if include_stderr == "WithStderr":
            std_err_prop = np.sqrt(prop*(1-prop)/num_sims)
            std_err_prop = np.round(std_err_prop,3)
            next_cell = "{} ({})".format(round(prop,3), std_err_prop)
            std_err_t1 = np.sqrt(t1_err*(1-t1_err)/num_sims)
            std_err_t1 = np.round(std_err_t1,3)
            next_cell += " {} ({})".format(round(t1_err,3), std_err_t1)
        else:
            next_cell = "{} {}".format(round(prop,3), round(t1_err,3))
    if outcome == OUTCOME_C:
        t1_err = np.sum(bin_curr["wald_reject"]) / num_sims      
#        t1_err = np.round(t1_err, 3)
        if include_stderr == "WithStderr":
            std_err_t1 = np.sqrt(t1_err*(1-t1_err)/num_sims)
            std_err_t1 = np.round(std_err_t1,3)
            next_cell = "{} ({})".format(round(t1_err,3), std_err_t1)
        else:
            next_cell = "{}".format(round(t1_err,3))         

    if outcome == "mean":
        next_cell = np.round(np.mean(df["imba"]), 3)
    if outcome == "std":
        next_cell = np.round(np.std(df["imba"]), 3)
    if outcome == "t1total":
        if include_stderr == "WithStderr":
            next_cell = t1_total 
        else:
            std_err_t1_total = np.sqrt(t1_total*(1-t1_total)/num_sims)
            next_cell = "{} ({})".format(round(t1_total,3), round(std_err_t1_total,3))


    return next_cell

def bin_abs_diff_apply_outcome(df = None, upper = 1.0, lower = 0.0, outcome = "Proportions", include_stderr = False):
    num_sims = len(df)
    df["abs_diff"] = np.abs(df["mean_1"] - df["mean_2"])
    df["wald_reject"] = df["wald_pval"] < 0.05
    bin_curr = df[(lower <= df["abs_diff"]) & (df["abs_diff"] < upper)]
    t1_total = np.round(np.sum(df["wald_reject"])/num_sims, 3)

    if outcome == OUTCOME_A:
        prop = len(bin_curr)/num_sims
        t1_err = np.sum(bin_curr["wald_reject"]) / num_sims
#        t1_err = np.round(t1_err, 3)

        if include_stderr == "WithStderr":
            std_err_prop = np.sqrt(prop*(1-prop)/num_sims)
            std_err_prop = np.round(std_err_prop,3)
            next_cell = "{} ({})".format(prop, std_err_prop)
            std_err_t1 = np.sqrt(t1_err*(1-t1_err)/num_sims)
            std_err_t1 = np.round(std_err_t1,3)
            next_cell += " {} ({})".format(t1_err, std_err_t1)
        else:
            next_cell = "{} {}".format(np.round(prop,3), np.round(t1_err,3))

    if outcome == OUTCOME_B:
        t1_err = np.sum(bin_curr["wald_reject"]) / num_sims      
#        t1_err = np.round(t1_err, 3)
        if include_stderr == "WithStderr":
            std_err_t1 = np.sqrt(t1_err*(1-t1_err)/num_sims)
            std_err_t1 = np.round(std_err_t1,3)
            next_cell = "{} ({})".format(round(t1_err,3), std_err_t1)
        else:
            next_cell = "{}".format(round(t1_err,3))         

    if outcome == "mean":
        next_cell = np.round(np.mean(df["abs_diff"]), 3)
    if outcome == "std":
        next_cell = np.round(np.std(df["abs_diff"]), 3)
    if outcome == "t1total":
        if include_stderr == "WithStderr":
            next_cell = t1_total 
        else:
            std_err_t1_total = np.round(np.sqrt(t1_total*(1-t1_total)/num_sims), 3)
            next_cell = "{} ({})".format(t1_total, std_err_t1_total)

    return next_cell


def set_bins(df = None, lower_bound = 0,  upper_bound = 1.0, step = 0.1, outcome = "Proportions", include_stderr = "NoStderr"):
    '''
    set bins for a row
    '''
    next_row = []
#    ipdb.set_trace()
    bins = np.round(np.arange(lower_bound, upper_bound, step), 3)
    col_header = []
    if outcome.split("-")[0].strip("Table") in "AB":
        mean_cell = bin_abs_diff_apply_outcome(df, outcome = "mean")
        var_cell = bin_abs_diff_apply_outcome(df, outcome = "std")
        t1total_cell = bin_abs_diff_apply_outcome(df, outcome = "t1total")

    elif outcome.split("-")[0].strip("Table") in "CD":
        mean_cell = bin_imba_apply_outcome(df, outcome = "mean")
        var_cell =  bin_imba_apply_outcome(df, outcome = "std")
        t1total_cell = bin_imba_apply_outcome(df, outcome = "t1total")

    next_row.append(mean_cell)
    next_row.append(var_cell)
    next_row.append(t1total_cell)
    col_header.append("Mean")
    col_header.append("Std")
    col_header.append("Type 1 Error Total")

    for lower in bins:
        upper = np.round(lower + step, 3) 
        
        if outcome.split("-")[0].strip("Table") in "AB":
            next_cell = bin_abs_diff_apply_outcome(df, upper, lower, outcome = outcome, include_stderr = include_stderr)
        
            col_header.append("[{}, {})".format(lower, upper))
        elif outcome.split("-")[0].strip("Table") in "CD":
            next_cell = bin_imba_apply_outcome(df, upper, lower, outcome = outcome, include_stderr = include_stderr)

            col_header.append("[{} %, {} %)".format(round(100*lower,2), round(100*upper,2)))#percentage
        next_row.append(next_cell)
#        col_header.append("[{}, {})".format(lower, upper))

    if outcome.split("-")[0].strip("Table") in "CD":
        next_cell = bin_imba_apply_outcome(df, 0.51, 0.48, outcome = outcome, include_stderr = include_stderr)
        next_row.append(next_cell)
        col_header.append("[48 %, 50 %]")
    return next_row, col_header

def set_rows_and_table(num_steps, df_alg_list, df_alg_key_list, outcome = "Proportions", iseffect = "NoEffect", include_stderr = "NoStderr", upper_bound = 1.0, lower_bound = 0.0, num_sims = 5000):
    '''
    Loop over algs, one for each row
    '''
    table_dict = {}
#    num_sims = len(df_alg_list[0])
    for df_alg, df_alg_key in zip(df_alg_list, df_alg_key_list):
        next_row = set_bins(df = df_alg, outcome = outcome, include_stderr = include_stderr, upper_bound = upper_bound, lower_bound = lower_bound)
        table_dict[df_alg_key], col_header = next_row

    table_df = pd.DataFrame(table_dict) 
    table_df.index = col_header 
    table_df = table_df.T

    save_dir = "../simulation_analysis_saves/Tables/{}/{}/{}/num_sims={}/".format(outcome, iseffect, include_stderr, num_sims)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_file = save_dir + "{}_n={}_numsims={}.csv".format(outcome, num_steps, num_sims)

    table_df.to_csv(save_file) 


def table_means_diff(df_ts = None, df_eg0pt1 = None, df_eg0pt3 = None, df_unif = None, n = None, \
                     title = None, iseffect = "NoEffect", num_sims = 5000):
    '''
    ''' 

    fig, ax = plt.subplots(2,2)       
    fig.set_size_inches(14.5, 10.5)
    ax = ax.ravel()
    i = 0                               
    
    step_sizes = df_unif['num_steps'].unique()
    size_vars = ["n/2", "n", "2*n", "4*n"]
    

    for num_steps in step_sizes:
   
        
        df_for_num_steps_eg0pt1 = df_eg0pt1[df_eg0pt1['num_steps'] == num_steps]
        df_for_num_steps_eg0pt3 = df_eg0pt3[df_eg0pt3['num_steps'] == num_steps]
        df_for_num_steps_unif = df_unif[df_unif['num_steps'] == num_steps]
        df_for_num_steps_ts = df_ts[df_ts['num_steps'] == num_steps]

        df_alg_list = [df_for_num_steps_ts, df_for_num_steps_eg0pt1, df_for_num_steps_eg0pt3, df_for_num_steps_unif]
        df_alg_list = [df_for_num_steps_unif, df_for_num_steps_ts, df_for_num_steps_eg0pt1, df_for_num_steps_eg0pt3]
        df_alg_key_list = ["Uniform", "Thompson Sampling", "Epsilon Greedy 0.1", "Epsilon Greedy 0.3"]

        include_stderr  = "WithStderr"

        set_rows_and_table(num_steps, df_alg_list, df_alg_key_list, outcome = OUTCOME_A, iseffect = iseffect, include_stderr = include_stderr, upper_bound = 1.0, lower_bound = 0.0, num_sims = num_sims) 
        set_rows_and_table(num_steps, df_alg_list, df_alg_key_list, outcome = OUTCOME_B, iseffect = iseffect, include_stderr = include_stderr, upper_bound = 1.0, lower_bound = 0.0, num_sims = num_sims) 
        set_rows_and_table(num_steps, df_alg_list, df_alg_key_list, outcome = OUTCOME_C, iseffect = iseffect, include_stderr = include_stderr, upper_bound = 0.5, lower_bound = 0.0, num_sims = num_sims) 
        set_rows_and_table(num_steps, df_alg_list, df_alg_key_list, outcome = OUTCOME_D, iseffect = iseffect, include_stderr = include_stderr, upper_bound = 0.5, lower_bound = 0.0, num_sims = num_sims) 


        include_stderr  = "NoStderr"

        set_rows_and_table(num_steps, df_alg_list, df_alg_key_list, outcome = OUTCOME_A, iseffect = iseffect, include_stderr = include_stderr, upper_bound = 1.0, lower_bound = 0.0, num_sims = num_sims) 
        set_rows_and_table(num_steps, df_alg_list, df_alg_key_list, outcome = OUTCOME_B, iseffect = iseffect, include_stderr = include_stderr, upper_bound = 1.0, lower_bound = 0.0, num_sims = num_sims) 
        set_rows_and_table(num_steps, df_alg_list, df_alg_key_list, outcome = OUTCOME_C, iseffect = iseffect, include_stderr = include_stderr, upper_bound = 0.5, lower_bound = 0.0, num_sims = num_sims) 
        set_rows_and_table(num_steps, df_alg_list, df_alg_key_list, outcome = OUTCOME_D, iseffect = iseffect, include_stderr = include_stderr, upper_bound = 0.5, lower_bound = 0.0, num_sims = num_sims) 




def convert_table3_dict_totable_loop(table_3_dict, c_list, n_list, num_sims, es_list, metric, c_list_labels, keys_list):
    #keys should be unfrmatted
    i = 0
    for n in n_list:# 
        step_sizes = [int(np.ceil(n/2)), n, 2*n, 4*n]
        es = es_list[i]
        i += 1
#        ipdb.set_trace()
        for num_steps in step_sizes: #One table per num_steps
            table_curr = []
            for c in c_list:
                col_curr = []
                for key in keys_list:
                    table_3_key = key.format(c, n, num_steps)
                    t1 = table_3_dict[table_3_key] 
                    col_curr.append(t1)
                col_curr = np.array(col_curr)

#                table_3_key_UR = TABLE_3_KEY_UR.format(c, n, num_steps)
#                table_3_key_TS = TABLE_3_KEY_TS.format(c, n, num_steps)
#                table_3_key_TSPPD = TABLE_3_KEY_TSPPD.format(c, n, num_steps)
#                table_3_key_ETS = TABLE_3_KEY_ETS.format(c, n, num_steps)
#
#                t1_UR = table_3_dict[table_3_key_UR] 
#                t1_TS = table_3_dict[table_3_key_TS] 
#                t1_TSPPD = table_3_dict[table_3_key_TSPPD]
#                t1_ETS = table_3_dict[table_3_key_ETS]
#
#                col_curr.append(t1_UR)
#                col_curr.append(t1_TS) 
#                col_curr.append(t1_TSPPD) 
#                col_curr.append(t1_ETS) 

                std_err_t1= np.round(np.sqrt(col_curr*(1-col_curr)/num_sims), 3) #to be consistent, need 1.96!!

                if metric == "Reward" or metric == "PropOpt":
                    std_err_t1 = []

                    for key in keys_list:
                    #std_err_t1 = []
                        table_3_key_var = key.format(c, n, num_steps) + "VAR"
                        t1_var = np.sqrt(table_3_dict[table_3_key_var])/np.sqrt(num_sims)
                        std_err_t1.append(np.round(t1_var, 3)) 

#                    table_3_key_UR_var = TABLE_3_KEY_UR.format(c, n, num_steps) + "VAR"
#                    table_3_key_TS_var = TABLE_3_KEY_TS.format(c, n, num_steps) + "VAR"
#                    table_3_key_TSPPD_var = TABLE_3_KEY_TSPPD.format(c, n, num_steps) + "VAR"
#                    table_3_key_ETS_var = TABLE_3_KEY_ETS.format(c, n, num_steps) + "VAR"
#
#                    #ipdb.set_trace()
#                    t1_UR_var = 1.96*np.sqrt(table_3_dict[table_3_key_UR_var])/np.sqrt(num_sims)
#                    t1_TS_var = 1.96*np.sqrt(table_3_dict[table_3_key_TS_var])/np.sqrt(num_sims)
#                    t1_TSPPD_var = 1.96*np.sqrt(table_3_dict[table_3_key_TSPPD_var])/np.sqrt(num_sims)
#                    t1_ETS_var = 1.96*np.sqrt(table_3_dict[table_3_key_ETS_var])/np.sqrt(num_sims)
#
#                    std_err_t1.append(np.round(t1_UR_var, 3))
#                    std_err_t1.append(np.round(t1_TS_var, 3)) 
#                    std_err_t1.append(np.round(t1_TSPPD_var, 3)) 
#                    std_err_t1.append(np.round(t1_ETS_var, 3)) 

                col_curr = np.round(col_curr, 3)
                next_cell = ["{} ({})".format(col_curr[i], std_err_t1[i]) for i in range(len(col_curr))]
                table_curr.append(next_cell)

            index = c_list_labels 
            columns = ["UR", "TS", "TSPPD", "ETS", "PPDGreedy"]
            columns = keys_list
            


            table_df = pd.DataFrame(table_curr, columns = columns, index = index) 
#            ipdb.set_trace()
            save_dir = "../simulation_analysis_saves/num_sims={}/Tables/Table_3_www/{}/N={}es={}/".format(num_sims, metric, n, es)
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_file = save_dir + "/Table_3_WWW_num_steps={}.csv".format(num_steps)
            print("Savig Table 3 WWW to,", save_file)
            table_df.to_csv(save_file)

def convert_table3_dict_totable(table_3_dict, c_list, n_list, num_sims, es_list, metric, c_list_labels):
    i = 0
    for n in n_list:# 
        step_sizes = [int(np.ceil(n/2)), n, 2*n, 4*n]
        es = es_list[i]
        i += 1
        for num_steps in step_sizes: #One table per num_steps
            table_curr = []
            for c in c_list:
                col_curr = []
                table_3_key_UR = TABLE_3_KEY_UR.format(c, n, num_steps)
                table_3_key_TS = TABLE_3_KEY_TS.format(c, n, num_steps)
                table_3_key_TSPPD = TABLE_3_KEY_TSPPD.format(c, n, num_steps)
                table_3_key_ETS = TABLE_3_KEY_ETS.format(c, n, num_steps)

                t1_UR = table_3_dict[table_3_key_UR] 
                t1_TS = table_3_dict[table_3_key_TS] 
                t1_TSPPD = table_3_dict[table_3_key_TSPPD]
                t1_ETS = table_3_dict[table_3_key_ETS]

                col_curr.append(t1_UR)
                col_curr.append(t1_TS) 
                col_curr.append(t1_TSPPD) 
                col_curr.append(t1_ETS) 

                col_curr = np.array(col_curr)
                std_err_t1= np.round(np.sqrt(col_curr*(1-col_curr)/num_sims), 3) #to be consistent, need 1.96!!

                if metric == "Reward" or metric == "PropOpt":
                    std_err_t1 = []
                    table_3_key_UR_var = TABLE_3_KEY_UR.format(c, n, num_steps) + "VAR"
                    table_3_key_TS_var = TABLE_3_KEY_TS.format(c, n, num_steps) + "VAR"
                    table_3_key_TSPPD_var = TABLE_3_KEY_TSPPD.format(c, n, num_steps) + "VAR"
                    table_3_key_ETS_var = TABLE_3_KEY_ETS.format(c, n, num_steps) + "VAR"

                    #ipdb.set_trace()
                    t1_UR_var = 1.96*np.sqrt(table_3_dict[table_3_key_UR_var])/np.sqrt(num_sims)
                    t1_TS_var = 1.96*np.sqrt(table_3_dict[table_3_key_TS_var])/np.sqrt(num_sims)
                    t1_TSPPD_var = 1.96*np.sqrt(table_3_dict[table_3_key_TSPPD_var])/np.sqrt(num_sims)
                    t1_ETS_var = 1.96*np.sqrt(table_3_dict[table_3_key_ETS_var])/np.sqrt(num_sims)

                    std_err_t1.append(np.round(t1_UR_var, 3))
                    std_err_t1.append(np.round(t1_TS_var, 3)) 
                    std_err_t1.append(np.round(t1_TSPPD_var, 3)) 
                    std_err_t1.append(np.round(t1_ETS_var, 3)) 

                col_curr = np.round(col_curr, 3)
                next_cell = ["{} ({})".format(col_curr[i], std_err_t1[i]) for i in range(len(col_curr))]
                table_curr.append(next_cell)

            index = c_list_labels 
            columns = ["UR", "TS", "TSPPD", "ETS"]


            table_df = pd.DataFrame(table_curr, columns = columns, index = index) 
#            ipdb.set_trace()
            save_dir = "../simulation_analysis_saves/num_sims={}/Tables/Table_3_www/{}/N={}es={}/".format(num_sims, metric, n, es)
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_file = save_dir + "/Table_3_WWW_num_steps={}.csv".format(num_steps)
            print("Savig Table 3 WWW to,", save_file)
            table_df.to_csv(save_file)


def www_table33_from_list(table_3_dict, c, n, num_steps, t1_list, keys_list):
    '''
    pass in paralel lists, [t1_ur,..] ; [TABLE_3_KEY_UR.format(..),..]
    '''

#    ipdb.set_trace()
    for t1, key in zip(t1_list, keys_list):
        table_3_dict[key] = t1

def www_table33(table_3_dict, c, n, num_steps, t1_unif, t1_ts, t1_tsppd_rs, t1_ets):
    assert t1_unif != None
    assert t1_ts != None
    assert t1_tsppd_rs != None
    assert t1_ets != None

    table_3_key_UR = TABLE_3_KEY_UR.format(c, n, num_steps)
    table_3_key_TS = TABLE_3_KEY_TS.format(c, n, num_steps)
    table_3_key_TSPPD = TABLE_3_KEY_TSPPD.format(c, n, num_steps)
    table_3_key_ETS = TABLE_3_KEY_ETS.format(c, n, num_steps)
    #ipdb.set_trace()

    table_3_dict[table_3_key_UR] = t1_unif 
    table_3_dict[table_3_key_TS] = t1_ts 
    table_3_dict[table_3_key_TSPPD] = t1_tsppd_rs 
    table_3_dict[table_3_key_ETS] = t1_ets 


#def www_table33_se_from_list(table_3_dict, c, n, num_steps, t1_unif_var, t1_ts_var, t1_tsppd_rs_var, t1_ets_var):
def www_table33_se_from_list(table_3_dict, c, n, num_steps, t1_var_list, keys_list):

#    ipdb.set_trace()
    for t1_var, key in zip(t1_var_list, keys_list):
        table_3_dict[key + "VAR"] = t1_var

def www_table33_se(table_3_dict, c, n, num_steps, t1_unif_var, t1_ts_var, t1_tsppd_rs_var, t1_ets_var):

    table_3_key_UR = TABLE_3_KEY_UR.format(c, n, num_steps) + "VAR"
    table_3_key_TS = TABLE_3_KEY_TS.format(c, n, num_steps) + "VAR"  
    table_3_key_TSPPD = TABLE_3_KEY_TSPPD.format(c, n, num_steps) + "VAR"
    table_3_key_ETS = TABLE_3_KEY_ETS.format(c, n, num_steps) + "VAR"
    #ipdb.set_trace()

    table_3_dict[table_3_key_UR] = t1_unif_var 
    table_3_dict[table_3_key_TS] = t1_ts_var 
    table_3_dict[table_3_key_TSPPD] = t1_tsppd_rs_var 
    table_3_dict[table_3_key_ETS] = t1_ets_var 

#    fig.suptitle(title)
#    #fig.tight_layout(rect=[0, 0.03, 1, 0.90])
#      # if not os.path.isdir("plots"):
#      #    os.path.mkdir("plots")
#    save_str_ne = "diff_hist/NoEffect/{}.png".format(title) 
#    save_str_e = "diff_hist/Effect/{}.png".format(title) 
#    if "No Effect" in title:
#	    print("saving to ", save_str_ne)
#	    fig.savefig(save_str_ne)
#    elif "With Effect" in title:
#	    print("saving to ", save_str_e)
#	    fig.savefig(save_str_e)
#
#      #plt.show()
#    plt.clf()
#    plt.close()





