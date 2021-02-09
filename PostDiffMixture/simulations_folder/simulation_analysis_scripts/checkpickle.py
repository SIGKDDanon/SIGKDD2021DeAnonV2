import matplotlib
matplotlib.use('Agg')
#matplotlib.use("gtk")
import matplotlib as mpl

#matplotlib.use('Qt5Agg')
from table_functions import *
import pickle
import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import sys
#sys.path.insert()
# print(data)
import numpy as np
import os
from scipy import stats
from matplotlib.pyplot import figure
import glob
import numpy as np
from hist_functions import *
import scipy.stats
from pathlib import Path
       # ipdb.set_trace()
import ipdb
from scatter_plot_functions import *
from rectify_vars_and_wald_functions import *
from phi_functions import *

TABLE_3_KEY_UR = "UR-c={}n={}num_steps={}"
TABLE_3_KEY_TS = "TS-c={}n={}num_steps={}"
TABLE_3_KEY_TSPPD = "TSPPD-c={}n={}num_steps={}"
TABLE_3_KEY_ETS = "ETS-c={}n={}num_steps={}"
TABLE_3_KEY_PPDG = "PPDG-c={}n={}num_steps={}"
TABLE_3_KEY_EG = "EG-c={}n={}num_steps={}"

SMALL_SIZE = 13
MEDIUM_SIZE = 10
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=8.5)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



def plot_hist_and_table(df_for_num_steps_eg0pt1, df_for_num_steps_eg0pt3, df_for_num_steps_ts, df_for_num_steps_unif, num_steps, epsilon, n):
        fig_h, ax_h = plt.subplots()
        proportions_unif = df_for_num_steps_unif['sample_size_1'] / num_steps
        proportions_eg0pt1 = df_for_num_steps_eg0pt1['sample_size_1'] / num_steps
        proportions_eg0pt3 = df_for_num_steps_eg0pt3['sample_size_1'] / num_steps
        proportions_ts = df_for_num_steps_ts['sample_size_1'] / num_steps
        
        ax_h.hist(proportions_eg0pt1, alpha = 0.5, label = "Epsilon Greedy 0.1")
        ax_h.hist(proportions_eg0pt3, alpha = 0.5, label = "Epsilon Greedy 0.3")
        ax_h.hist(proportions_unif, alpha = 0.5, label = "Uniform Random")
        ax_h.hist(proportions_ts, alpha = 0.5, label = "Thompson Sampling")
        ax_h.legend()
        fig_h.suptitle("Histogram of Proportion of {} Participants Assigned to Condition 1 Across 500 Simulations".format(num_steps))
       # rows = ["Areferg"]
       # columns = ["Berger"]
       # cell_text = ["ergerg"]
       # the_table = ax_h.table(cellText=cell_text,
         #             rowLabels=rows,
        #              colLabels=columns,
          #            loc='right')

      #  fig_h.subplots_adjust(left=0.2, wspace=0.4)
        data = np.random.uniform(0, 1, 80).reshape(20, 4)
        mean_ts = np.mean(proportions_ts)
        var_ts = np.var(proportions_ts)

        mean_eg0pt1 = np.mean(proportions_eg0pt1)
        mean_eg0pt3 = np.mean(proportions_eg0pt3)
        var_eg0pt1 = np.var(proportions_eg0pt1)
        var_eg0pt3 = np.var(proportions_eg0pt3)

        prop_lt_25_eg0pt1 = np.sum(proportions_eg0pt1 < 0.25) / len(proportions_eg0pt1)
        prop_lt_25_eg0pt3 = np.sum(proportions_eg0pt3 < 0.25) / len(proportions_eg0pt3)
        prop_lt_25_ts = np.sum(proportions_ts < 0.25) / len(proportions_ts)

       # prop_gt_25_lt_5_eg = np.sum(> proportions > 0.25) / len(proportions)
       # prop_gt_25_lt_5_ts = np.sum(> proportions_ts > 0.25) / len(proportions_ts)

        data = [[mean_ts, var_ts, prop_lt_25_ts],\
         [mean_eg0pt1, var_eg0pt1, prop_lt_25_eg0pt1],\
         [mean_eg0pt3, var_eg0pt3, prop_lt_25_eg0pt3]]


        final_data = [['%.3f' % j for j in i] for i in data] #<0.25, 0.25< & <0.5, <0.5 & <0.75, <0.75 & <1.0                                                                                                                 
        #table.auto_set_font_size(False)
      #  table.set_fontsize(7)
      #  table.auto_set_column_width((-1, 0, 1, 2, 3))
        table = ax_h.table(cellText=final_data, colLabels=['Mean', 'Variance', 'prop < 0.25'], rowLabels = ["Thompson Sampling", "Epsilon Greedy 0.1", "Epsilon Greedy 0.3"], loc='bottom', cellLoc='center', bbox=[0.25, -0.5, 0.5, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.auto_set_column_width((-1, 0, 1, 2, 3))

        # Adjust layout to make room for the table:
        #ax_h.tick_params(axis='x', pad=20)

        #fig_h.subplots_adjust(left=0.2, bottom=0.5)
        #fig_h.tight_layout()
        save_dir = "../simulation_analysis_saves/histograms/ExploreAndExploit/N={}".format(n)
        Path(save_dir).mkdir(parents=True, exist_ok=True)


        fig_h.savefig(save_dir + "/condition_prop_n={}.png".format(num_steps), bbox_inches = 'tight')
        fig_h.clf()




PROP_EPS_KEY = "prop_exploring_ppd_cuml"


def compute_propeps_var(df_for_num_steps, num_steps):
     mean_prop_eps = df_for_num_steps[PROP_EPS_KEY].var()
     return mean_prop_eps

def compute_propeps(df_for_num_steps, num_steps):
     mean_prop_eps = np.mean(df_for_num_steps[PROP_EPS_KEY])
     return mean_prop_eps

def compute_propmaj(df_for_num_steps, num_steps):

     propmaj_1_df =  df_for_num_steps[df_for_num_steps['mean_1'] >= df_for_num_steps['mean_2']]['sample_size_1']/num_steps
     propmaj_2_df =  df_for_num_steps[df_for_num_steps['mean_2'] > df_for_num_steps['mean_1']]['sample_size_2']/num_steps
     comb = propmaj_1_df.append(propmaj_2_df)
#     ipdb.set_trace()
     assert(len(comb) == 10000)
#
     return comb.mean()
     #return (propmaj_1 + propmaj_2)/2

def compute_propmaj_var(df_for_num_steps, num_steps):

     propmaj_1_df =  df_for_num_steps[df_for_num_steps['mean_1'] > df_for_num_steps['mean_2']]['sample_size_1']/num_steps
     propmaj_2_df =  df_for_num_steps[df_for_num_steps['mean_2'] > df_for_num_steps['mean_1']]['sample_size_2']/num_steps
     comb = propmaj_1_df.append(propmaj_2_df)
     assert(len(comb) == 10000)
#     ipdb.set_trace()

     return comb.var()

def compute_propopt_var_single(df_for_num_steps, num_steps):
     propopt_var = (df_for_num_steps['sample_size_1']/num_steps).var()
     return propopt_var

def compute_propopt_var(df_for_num_steps_eg0pt1,df_for_num_steps_eg0pt3, df_for_num_steps_unif, df_for_num_steps_ts, df_for_num_steps_tsppd,
                      df_for_num_steps_ets, df_for_num_steps_tsppd_rs, num_steps): 

     unif_reward_mean = (df_for_num_steps_unif['sample_size_1']/num_steps).var()
     ts_reward_mean = (df_for_num_steps_ts['sample_size_1']/num_steps).var()
     eps_greedy_reward_mean_0pt1 = (df_for_num_steps_eg0pt1['sample_size_1']/num_steps).var()
     eps_greedy_reward_mean_0pt3 = (df_for_num_steps_eg0pt3['sample_size_1']/num_steps).var()
 
     tsppd_reward_mean = (df_for_num_steps_tsppd['sample_size_1']/num_steps).var()
     tsppd_rs_reward_mean = (df_for_num_steps_tsppd_rs['sample_size_1']/num_steps).var()
     ets_reward_mean = (df_for_num_steps_ets['sample_size_1']/num_steps).var()

     return eps_greedy_reward_mean_0pt1, eps_greedy_reward_mean_0pt3, ts_reward_mean, tsppd_reward_mean, ets_reward_mean, unif_reward_mean, tsppd_rs_reward_mean

def compute_propopt_single(df_for_num_steps, num_steps):
     propopt_mean = (df_for_num_steps['sample_size_1']/num_steps).mean()
     return propopt_mean

def compute_propopt(df_for_num_steps_eg0pt1,df_for_num_steps_eg0pt3, df_for_num_steps_unif, df_for_num_steps_ts, df_for_num_steps_tsppd,
                      df_for_num_steps_ets, df_for_num_steps_tsppd_rs, num_steps): 

     unif_reward_mean = (df_for_num_steps_unif['sample_size_1']/num_steps).mean()
     ts_reward_mean = (df_for_num_steps_ts['sample_size_1']/num_steps).mean()
     eps_greedy_reward_mean_0pt1 = (df_for_num_steps_eg0pt1['sample_size_1']/num_steps).mean()
     eps_greedy_reward_mean_0pt3 = (df_for_num_steps_eg0pt3['sample_size_1']/num_steps).mean()
 
     tsppd_reward_mean = (df_for_num_steps_tsppd['sample_size_1']/num_steps).mean()
     tsppd_rs_reward_mean = (df_for_num_steps_tsppd_rs['sample_size_1']/num_steps).mean()
     ets_reward_mean = (df_for_num_steps_ets['sample_size_1']/num_steps).mean()

     return eps_greedy_reward_mean_0pt1, eps_greedy_reward_mean_0pt3, ts_reward_mean, tsppd_reward_mean, ets_reward_mean, unif_reward_mean, tsppd_rs_reward_mean

def compute_reward_var_single(df_for_num_steps, num_steps):
     reward_var = (df_for_num_steps['total_reward']/num_steps).var()

     return reward_var
    
def compute_reward_var(df_for_num_steps_eg0pt1,df_for_num_steps_eg0pt3, df_for_num_steps_unif, df_for_num_steps_ts, df_for_num_steps_tsppd,
                      df_for_num_steps_ets, df_for_num_steps_tsppd_rs, num_steps): 

     unif_reward_mean = (df_for_num_steps_unif['total_reward']/num_steps).var()
     ts_reward_mean = (df_for_num_steps_ts['total_reward']/num_steps).var()
     eps_greedy_reward_mean_0pt1 = (df_for_num_steps_eg0pt1['total_reward']/num_steps).var()
     eps_greedy_reward_mean_0pt3 = (df_for_num_steps_eg0pt3['total_reward']/num_steps).var()
 
     tsppd_reward_mean = (df_for_num_steps_tsppd['total_reward']/num_steps).var()
     tsppd_rs_reward_mean = (df_for_num_steps_tsppd_rs['total_reward']/num_steps).var()
     ets_reward_mean = (df_for_num_steps_ets['total_reward']/num_steps).var()

     return eps_greedy_reward_mean_0pt1, eps_greedy_reward_mean_0pt3, ts_reward_mean, tsppd_reward_mean, ets_reward_mean, unif_reward_mean, tsppd_rs_reward_mean

def compute_reward_single(df_for_num_steps, num_steps):
     reward_mean = (df_for_num_steps['total_reward']/num_steps).mean()
     return reward_mean

def compute_reward(df_for_num_steps_eg0pt1,df_for_num_steps_eg0pt3, df_for_num_steps_unif, df_for_num_steps_ts, df_for_num_steps_tsppd,
                      df_for_num_steps_ets, df_for_num_steps_tsppd_rs, num_steps): 

     unif_reward_mean = (df_for_num_steps_unif['total_reward']/num_steps).mean()
     ts_reward_mean = (df_for_num_steps_ts['total_reward']/num_steps).mean()
     eps_greedy_reward_mean_0pt1 = (df_for_num_steps_eg0pt1['total_reward']/num_steps).mean()
     eps_greedy_reward_mean_0pt3 = (df_for_num_steps_eg0pt3['total_reward']/num_steps).mean()
 
     tsppd_reward_mean = (df_for_num_steps_tsppd['total_reward']/num_steps).mean()
     tsppd_rs_reward_mean = (df_for_num_steps_tsppd_rs['total_reward']/num_steps).mean()
     ets_reward_mean = (df_for_num_steps_ets['total_reward']/num_steps).mean()

     return eps_greedy_reward_mean_0pt1, eps_greedy_reward_mean_0pt3, ts_reward_mean, tsppd_reward_mean, ets_reward_mean, unif_reward_mean, tsppd_rs_reward_mean

#t1_eg0pt1, t1_eg0pt3, t1_ts, t1_tsppd, t1_ets, t1_unif, var
def compute_power_single(df_for_num_steps):

    num_replications = len(df_for_num_steps)
    num_rejected = np.sum(df_for_num_steps['wald_pval'] < .05) #
    t1= num_rejected / num_replications

    return t1

def compute_power(df_for_num_steps_eg0pt1,df_for_num_steps_eg0pt3, df_for_num_steps_unif, df_for_num_steps_ts, df_for_num_steps_tsppd,
                          df_for_num_steps_ets, df_for_num_steps_tsppd_rs): 

       # ipdb.set_trace()
    num_replications = len(df_for_num_steps_eg0pt1)
        #num_rejected_eg0pt1 = np.sum(df_for_num_steps_eg0pt1['pvalue'] < .05) #Epsilon Greedy
    num_rejected_eg0pt1 = np.sum(df_for_num_steps_eg0pt1['wald_pval'] < .05) #Epsilon Greedy
        #num_rejected_eg0pt1 = np.sum(wald_pval_eg0pt1 < .05) #Epsilon Greedy

        #num_rejected_eg0pt3 = np.sum(df_for_num_steps_eg0pt3['pvalue'] < .05) #Epsilon Greedy
    num_rejected_eg0pt3 = np.sum(df_for_num_steps_eg0pt3['wald_pval'] < .05) #Epsilon Greedy

        #num_rejected_ts = np.sum(df_for_num_steps_ts['pvalue'] < .05) #Thompson
    num_rejected_ts = np.sum(df_for_num_steps_ts['wald_pval'] < .05) #Thompson
    num_rejected_tsppd = np.sum(df_for_num_steps_tsppd['wald_pval'] < .05) #Thompson
    num_rejected_tsppd_rs = np.sum(df_for_num_steps_tsppd_rs['wald_pval'] < .05) #Thompson
    num_rejected_ets = np.sum(df_for_num_steps_ets['wald_pval'] < .05) #Thompson

#        num_rejected_unif = np.sum(df_for_num_steps_unif['pvalue'] < .05)
    num_rejected_unif = np.sum(df_for_num_steps_unif['wald_pval'] < .05)

    var = np.var(df_for_num_steps_unif['pvalue'] < .05)
        
    num_replications = len(df_for_num_steps_eg0pt1)
    t1_eg0pt1 = num_rejected_eg0pt1 / num_replications
    num_replications = len(df_for_num_steps_eg0pt3)
    t1_eg0pt3 = num_rejected_eg0pt3 / num_replications

    num_replications = len(df_for_num_steps_ts)
    t1_ts = num_rejected_ts / num_replications

    num_replications = len(df_for_num_steps_tsppd)
    t1_tsppd = num_rejected_tsppd / num_replications

    num_replications = len(df_for_num_steps_tsppd_rs)
    t1_tsppd_rs = num_rejected_tsppd_rs / num_replications

    num_replications = len(df_for_num_steps_ets)
    t1_ets = num_rejected_ets / num_replications


    num_replications = len(df_for_num_steps_unif)
    t1_unif =num_rejected_unif / num_replications


    return t1_eg0pt1, t1_eg0pt3, t1_ts, t1_tsppd, t1_ets, t1_unif, t1_tsppd_rs, var

def open_df_list(model_dir_list, n):
    bs = 1

    df_list = []

    for model_dir in model_dir_list:

        to_check = glob.glob(model_dir + "/*Prior*{}*{}Df*.pkl".format(bs,n))[0] #Has eg, 34 in 348!!
        assert(len(glob.glob(model_dir + "/*Prior*{}*{}Df*.pkl".format(bs,n))) == 1)

        with open(to_check, 'rb') as f:
            df = pickle.load(f)

        rect_key = "Drop NA"
        rectify_vars_noNa(df, alg_key = rect_key)
        assert np.sum(df["wald_type_stat"].isna()) == 0

        df_list.append(df)

    return df_list


def process_model_dir_list(model_dir_list, model_key_list, c, n, table_3_dict, metric = "Power&T1", es = 0):

    #open df list
    if es == 0:
        df_list = open_df_list(model_dir_list, n)
    else:
        df_list = open_df_list(model_dir_list, es)
    #outer loop over numsteps
    step_sizes = df_list[0]['num_steps'].unique()

    for num_steps in step_sizes:
        t1_list = []
        keys_list_formatted = []
        t1_var_list = []
   #     ipdb.set_trace()
        i = 0
        for df in df_list:
            df_for_num_steps = df[df['num_steps'] == num_steps].dropna()
            t1_curr, t1_var_curr = get_metric_numsteps(df_for_num_steps, metric = metric)

            t1_list.append(t1_curr)
            t1_var_list.append(t1_var_curr)
            key_curr = model_key_list[i].format(c, n, num_steps)
            keys_list_formatted.append(key_curr)
            i+=1

#        pass in by list
        www_table33_from_list(table_3_dict = table_3_dict, c = c, n = n, num_steps = num_steps, t1_list = t1_list, keys_list = keys_list_formatted)

        if metric == "Reward" or "PropOpt" in metric:
            www_table33_se_from_list(table_3_dict = table_3_dict, c = c, n = n, num_steps = num_steps, t1_var_list = t1_var_list, keys_list = keys_list_formatted)

def process_model_dir(model_dir):
    '''
    return t1 for a given model
    '''
    bs = 1

    to_check = glob.glob(model_dir + "/*Prior*{}*{}Df*.pkl".format(bs,n))[0] #Has eg, 34 in 348!!
    assert(len(glob.glob(model_dir + "/*Prior*{}*{}Df*.pkl".format(bs,n))) == 1)

    with open(to_check, 'rb') as f:
        df = pickle.load(f)

    rect_key = "Drop NA"
    rectify_vars_noNa(df, alg_key = rect_key)
    assert np.sum(df["wald_type_stat"].isna()) == 0

    get_metric_numsteps(df)

def get_metric_numsteps(df_for_num_steps, metric):

 #    ipdb.set_trace()
    t1_var = -99
    #num_steps = df_for_num_steps["num_steps"].iloc[0]
    num_steps = df_for_num_steps["num_steps"].iloc[0]

    if metric == "Power&T1":     
        t1 = compute_power_single(df_for_num_steps) 
    elif metric == "Reward":
        t1 = compute_reward_single(df_for_num_steps, num_steps) 
        t1_var = compute_reward_var_single(df_for_num_steps, num_steps)

    elif metric == "PropOpt":
        t1 = compute_propopt_single(df_for_num_steps, num_steps) 
        t1_var = compute_propopt_var_single(df_for_num_steps, num_steps) 

    elif metric == "PropMaj":
        t1 = compute_propmaj(df_for_num_steps, num_steps) 
        t1_var = compute_propmaj_var(df_for_num_steps, num_steps) 

    elif metric == "PropEps":
        t1 = compute_propeps(df_for_num_steps, num_steps) 
        t1_var = compute_propeps_var(df_for_num_steps, num_steps) 

    return t1, t1_var

def stacked_bar_plot_with_cutoff(df_ts = None, df_eg0pt1 = None, df_eg0pt3 = None, df_unif = None, df_tsppd = None, n = None, num_sims = None, df_ets = None, df_tsppd_rs = None, df_ppdg = None,\
                     title = None, bs_prop = 0.0,\
                     ax = None, ax_idx = None, epsilon = None, c = None, ymax = None, hline = 0.05, metric = "Power&T1", c_idx = None, c_list = None, table_3_dict = None, es = 0):
    
    step_sizes = df_unif['num_steps'].unique()
    size_vars = ["n/2", "n", "2*n", "4*n"]
    t1_list_eg0pt1 = []
    t1_list_eg0pt3 = []
    
    t1_list_unif = []
    t1_wald_list_unif = []
    var_list = []
    t1_list_ts = []
    t1_list_tsppd = []
    t1_list_tsppd_rs = []
    t1_list_ets = []
    var = 0 #TODO change this
    for num_steps in step_sizes:
        df_for_num_steps_eg0pt1 = df_eg0pt1[df_eg0pt1['num_steps'] == num_steps].dropna()
        df_for_num_steps_eg0pt3 = df_eg0pt3[df_eg0pt3['num_steps'] == num_steps].dropna()
        df_for_num_steps_unif = df_unif[df_unif['num_steps'] == num_steps].dropna()
        df_for_num_steps_ts = df_ts[df_ts['num_steps'] == num_steps].dropna()
        df_for_num_steps_tsppd = df_tsppd[df_tsppd['num_steps'] == num_steps].dropna()
        df_for_num_steps_tsppd_rs = df_tsppd_rs[df_tsppd_rs['num_steps'] == num_steps].dropna()
        df_for_num_steps_ets = df_ets[df_ets['num_steps'] == num_steps].dropna()
        #df_for_num_steps_unif = df_for_num_steps_unif.dropna()
       # bins = np.arange(0, 1.01, .025)

   #     plot_hist_and_table(df_for_num_steps_eg0pt1, df_for_num_steps_eg0pt3, df_for_num_steps_ts, df_for_num_steps_unif, num_steps, epsilon = epsilon, n=n)


#        ipdb.set_trace()
        if metric == "Power&T1":     
            t1_eg0pt1, t1_eg0pt3, t1_ts, t1_tsppd, t1_ets, t1_unif, t1_tsppd_rs, var = compute_power(df_for_num_steps_eg0pt1,df_for_num_steps_eg0pt3, df_for_num_steps_unif, df_for_num_steps_ts, df_for_num_steps_tsppd, df_for_num_steps_ets, df_for_num_steps_tsppd_rs) 
        elif metric == "Reward":
            t1_eg0pt1, t1_eg0pt3, t1_ts, t1_tsppd, t1_ets, t1_unif, t1_tsppd_rs = compute_reward(df_for_num_steps_eg0pt1,df_for_num_steps_eg0pt3, df_for_num_steps_unif, df_for_num_steps_ts, df_for_num_steps_tsppd, df_for_num_steps_ets, df_for_num_steps_tsppd_rs, num_steps) 

            t1_eg0pt1_var, t1_eg0pt3_var, t1_ts_var, t1_tsppd_var, t1_ets_var, t1_unif_var, t1_tsppd_rs_var = compute_reward_var(df_for_num_steps_eg0pt1,df_for_num_steps_eg0pt3, df_for_num_steps_unif, df_for_num_steps_ts, df_for_num_steps_tsppd, df_for_num_steps_ets, df_for_num_steps_tsppd_rs, num_steps) 

        elif metric == "PropOpt" or metric == "PropEps":
            t1_eg0pt1, t1_eg0pt3, t1_ts, t1_tsppd, t1_ets, t1_unif, t1_tsppd_rs = compute_propopt(df_for_num_steps_eg0pt1,df_for_num_steps_eg0pt3, df_for_num_steps_unif, df_for_num_steps_ts, df_for_num_steps_tsppd, df_for_num_steps_ets, df_for_num_steps_tsppd_rs, num_steps) 
            t1_eg0pt1_var, t1_eg0pt3_var, t1_ts_var, t1_tsppd_var, t1_ets_var, t1_unif_var, t1_tsppd_rs_var = compute_propopt_var(df_for_num_steps_eg0pt1,df_for_num_steps_eg0pt3, df_for_num_steps_unif, df_for_num_steps_ts, df_for_num_steps_tsppd, df_for_num_steps_ets, df_for_num_steps_tsppd_rs, num_steps) 

        www_table33(table_3_dict = table_3_dict, c = c, n = n, num_steps = num_steps, t1_unif = t1_unif, t1_ts = t1_ts, t1_tsppd_rs = t1_tsppd_rs, t1_ets = t1_ets)

        if metric == "Reward" or "PropOpt" in metric:
            www_table33_se(table_3_dict = table_3_dict, c = c, n = n, num_steps = num_steps, t1_unif_var = t1_unif_var, t1_ts_var = t1_ts_var, t1_tsppd_rs_var = t1_tsppd_rs_var, t1_ets_var = t1_ets_var)
          #  table_3_key_UR = "UR-c={}n={}num_steps={}".format(c, n, num_steps)
      #  table_3_key_TS = "TS-c={}n={}num_steps={}".format(c, n, num_steps)
      #  table_3_key_TSPPD = "TSPPD-c={}n={}num_steps={}".format(c, n, num_steps)
      #  table_3_key_ETS = "ETS-c={}n={}num_steps={}".format(c, n, num_steps)
      #  table_3_dict[table_3_key_UR] = t1_unif 
      #  table_3_dict[table_3_key_TS] = t1_ts 
      #  table_3_dict[table_3_key_TSPPD] = t1_tsppd_rs 
      #  table_3_dict[table_3_key_ETS] = t1_ets 

        t1_list_unif.append(t1_unif)
        t1_list_ts.append(t1_ts)
        t1_list_tsppd.append(t1_tsppd)
        t1_list_tsppd_rs.append(t1_tsppd_rs)
        t1_list_ets.append(t1_ets)
        
        t1_list_eg0pt1.append(t1_eg0pt1)
        t1_list_eg0pt3.append(t1_eg0pt3)
        var_list.append(var)

        
    t1_list_ts = np.array(t1_list_ts)
    t1_list_tsppd = np.array(t1_list_tsppd)
    t1_list_tsppd_rs = np.array(t1_list_tsppd_rs)
    t1_list_ets = np.array(t1_list_ets)
    ind = np.arange(3*len(step_sizes), step=3)
 #   print(ind)
  #  print(step_sizes)
    ax.set_xticks(ind)
    ax.set_xticklabels(step_sizes)
   
#    print("var", var_list)
    width = 0.16
    capsize = width*4
    width_total = 2*width
    
     
    t1_list_eg0pt1 = np.array(t1_list_eg0pt1)
    t1_list_eg0pt3 = np.array(t1_list_eg0pt3)
    t1_list_unif = np.array(t1_list_unif)
    
    num_sims_RL4RL = 5000
    t1_eg0pt1_se = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_list_eg0pt1*(1-t1_list_eg0pt1)/num_sims_RL4RL) #95 CI for Proportion
    t1_eg0pt3_se = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_list_eg0pt3*(1-t1_list_eg0pt3)/num_sims_RL4RL) #95 CI for Proportion
   
    t1_se_unif = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_list_unif*(1-t1_list_unif)/num_sims_RL4RL)
    t1_se_ts = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_list_ts*(1-t1_list_ts)/num_sims_RL4RL)
    num_sims_ppd = 5000
    t1_se_tsppd = stats.t.ppf(1-0.025, num_sims_ppd)*np.sqrt(t1_list_tsppd*(1-t1_list_tsppd)/num_sims_ppd)
    t1_se_tsppd_rs = stats.t.ppf(1-0.025, num_sims_ppd)*np.sqrt(t1_list_tsppd_rs*(1-t1_list_tsppd_rs)/num_sims_ppd)

    num_sims_ets = 5000
    t1_se_ets = stats.t.ppf(1-0.025, num_sims_ppd)*np.sqrt(t1_list_ets*(1-t1_list_ets)/num_sims_ets)

#    print(t1_se_unif) #note that power goes to 1.0 for unif, thus error bars
    #print(t1_se_unif)
#    p1 = ax.bar(ind, t1_list_eg0pt1, width = width, yerr = t1_eg0pt1_se, \
 #               ecolor='black', capsize=capsize, color = 'yellow', edgecolor='black')
    
 #   p3 = ax.bar(ind+width, t1_list_eg0pt3, width = width, yerr = t1_eg0pt3_se, \
 #               ecolor='black', capsize=capsize, color = 'grey', edgecolor='black')
  
    recenter = 9*width
    p3 = ax.bar(ind + 10*width + (c_idx)*width - recenter, t1_list_tsppd_rs, width = width, yerr = t1_se_tsppd_rs,     
           ecolor='black', capsize=capsize, color = 'yellow', edgecolor='black') 

    p4 = ax.bar(ind+2*width - recenter, t1_list_ts, width = width, yerr = t1_se_ts,     
               ecolor='black', capsize=capsize, color = 'blue', edgecolor='black') 

   # p5 = ax.bar(ind+5*width + c_idx*width - recenter, t1_list_tsppd, width = width, yerr = t1_se_tsppd,     
   #            ecolor='black', capsize=capsize, color = 'purple', edgecolor='black') 

    p6 = ax.bar(ind+3*width + (c_idx)*width - recenter, t1_list_ets, width = width, yerr = t1_se_ets,     
               ecolor='black', capsize=capsize, color = 'brown', edgecolor='black') 

  
    p2 = ax.bar(ind + 11*width + 6*width - recenter, t1_list_unif, width = width,\
                yerr = t1_se_unif, ecolor='black', \
               capsize=capsize, color = 'red', \
               edgecolor='black')

    if ax_idx == 1 and c_idx != 0:
      # leg1 = ax.legend((p2[0], p1[0], p3[0], p4[0], p6[0], p5[0]), ("Uniform Wald", 'Epsilon Greedy 0.1 Wald', "Epsilon Greedy 0.3 Wald", "Thompson Sampling Wald", "Epsilon 0.1 Thompson Sampling Wald", "PostDiff Thompson Sampling Wald \n c left to right [0.08, 0.1, 0.12]"), bbox_to_anchor=(1.0, 1.76))  
       leg1 = ax.legend((p4[0], p6[0], p3[0], p2[0]), ( "Thompson Sampling Wald", "Epsilon Thompson Sampling Wald \n Epsilon left to right {}".format(c_list), "PostDiff Thompson Sampling Wald \n c left to right {}".format(c_list), "Uniform Random"), bbox_to_anchor=(1.0, 1.82))  
    #leg2 = ax.legend(loc = 2)
    
       ax.add_artist(leg1)
 #   plt.tight_layout()
   # plt.title(title)
#    if ax_idx == 6 or ax_idx == 7 or ax_idx == 8:
    ax.set_xlabel("number of participants = \n n/2, n, 2*n, 4*n")
    ax.set_ylim(0, ymax)
#    ax.set_ylim(0, 0.58)
    if hline != None:
        ax.axhline(y=hline, linestyle='--')


    return [t1_list_ts, t1_list_ets, t1_list_tsppd_rs, t1_list_unif] #returns [UR, ppd, TS, ets], in this case, need to return for each step size, but only plotting for one bs, so save step size by model (4x2)

def line_plot(df_ts = None, df_eg0pt1 = None, df_eg0pt3 = None, df_unif = None, df_tsppd = None, n = None, num_sims = None, df_ets = None, df_tsppd_rs = None,\
                     title = None, bs_prop = 0.0,\
                     ax = None, ax_idx = None, epsilon = None, c = None, ymax = None, hline = 0.05, metric = "Power&T1", c_idx = None):
    
    step_sizes = df_unif['num_steps'].unique()
    size_vars = ["n/2", "n", "2*n", "4*n"]
    c_idx_mapping= {0.08: 0, 0.1:1, 0.12:2}
   # c_idx = c_idx_mapping[c]
    t1_list_eg0pt1 = []
    t1_list_eg0pt3 = []
    
    t1_list_unif = []
    t1_wald_list_unif = []
    var_list = []
    t1_list_ts = []
    t1_list_tsppd = []
    t1_list_tsppd_rs = []
    t1_list_ets = []
    var = 0 #TODO change this
    for num_steps in step_sizes:
        
        df_for_num_steps_eg0pt1 = df_eg0pt1[df_eg0pt1['num_steps'] == num_steps].dropna()
        df_for_num_steps_eg0pt3 = df_eg0pt3[df_eg0pt3['num_steps'] == num_steps].dropna()
        df_for_num_steps_unif = df_unif[df_unif['num_steps'] == num_steps].dropna()
        df_for_num_steps_ts = df_ts[df_ts['num_steps'] == num_steps].dropna()
        df_for_num_steps_tsppd = df_tsppd[df_tsppd['num_steps'] == num_steps].dropna()
        df_for_num_steps_tsppd_rs = df_tsppd_rs[df_tsppd_rs['num_steps'] == num_steps].dropna()
        df_for_num_steps_ets = df_ets[df_ets['num_steps'] == num_steps].dropna()
        #df_for_num_steps_unif = df_for_num_steps_unif.dropna()
       # bins = np.arange(0, 1.01, .025)

   #     plot_hist_and_table(df_for_num_steps_eg0pt1, df_for_num_steps_eg0pt3, df_for_num_steps_ts, df_for_num_steps_unif, num_steps, epsilon = epsilon, n=n)


        if metric == "Power&T1":     
            t1_eg0pt1, t1_eg0pt3, t1_ts, t1_tsppd, t1_ets, t1_unif, t1_tsppd_rs, var = compute_power(df_for_num_steps_eg0pt1,df_for_num_steps_eg0pt3, df_for_num_steps_unif, df_for_num_steps_ts, df_for_num_steps_tsppd, df_for_num_steps_ets, df_for_num_steps_tsppd_rs) 
        elif metric == "Reward":
            t1_eg0pt1, t1_eg0pt3, t1_ts, t1_tsppd, t1_ets, t1_unif, t1_tsppd_rs = compute_reward(df_for_num_steps_eg0pt1,df_for_num_steps_eg0pt3, df_for_num_steps_unif, df_for_num_steps_ts, df_for_num_steps_tsppd, df_for_num_steps_ets, df_for_num_steps_tsppd_rs, num_steps) 

        t1_list_unif.append(t1_unif)
        t1_list_ts.append(t1_ts)
        t1_list_tsppd.append(t1_tsppd)
        t1_list_tsppd_rs.append(t1_tsppd_rs)
        t1_list_ets.append(t1_ets)
        
        t1_list_eg0pt1.append(t1_eg0pt1)
        t1_list_eg0pt3.append(t1_eg0pt3)
        var_list.append(var)

        
    t1_list_ts = np.array(t1_list_ts)
    t1_list_tsppd = np.array(t1_list_tsppd)
    t1_list_tsppd_rs = np.array(t1_list_tsppd_rs)
    t1_list_ets = np.array(t1_list_ets)
    ind = np.arange(3*len(step_sizes), step=3)
 #   print(ind)
  #  print(step_sizes)
    ax.set_xticks(ind)
    ax.set_xticklabels(step_sizes)
   
    width = 0.30
    capsize = width*4
    width_total = 2*width
    
     
    t1_list_eg0pt1 = np.array(t1_list_eg0pt1)
    t1_list_eg0pt3 = np.array(t1_list_eg0pt3)
    t1_list_unif = np.array(t1_list_unif)
    
    num_sims_RL4RL = 5000
    t1_eg0pt1_se = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_list_eg0pt1*(1-t1_list_eg0pt1)/num_sims_RL4RL) #95 CI for Proportion
    t1_eg0pt3_se = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_list_eg0pt3*(1-t1_list_eg0pt3)/num_sims_RL4RL) #95 CI for Proportion
   
    t1_se_unif = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_list_unif*(1-t1_list_unif)/num_sims_RL4RL)
    t1_se_ts = stats.t.ppf(1-0.025, num_sims)*np.sqrt(t1_list_ts*(1-t1_list_ts)/num_sims_RL4RL)
    num_sims_ppd = 5000
    t1_se_tsppd = stats.t.ppf(1-0.025, num_sims_ppd)*np.sqrt(t1_list_tsppd*(1-t1_list_tsppd)/num_sims_ppd)
    t1_se_tsppd_rs = stats.t.ppf(1-0.025, num_sims_ppd)*np.sqrt(t1_list_tsppd_rs*(1-t1_list_tsppd_rs)/num_sims_ppd)

    num_sims_ets = 5000
    t1_se_ets = stats.t.ppf(1-0.025, num_sims_ppd)*np.sqrt(t1_list_ets*(1-t1_list_ets)/num_sims_ets)

    n_lines = 7
    c_ = np.arange(1, n_lines + 1)

    norm = mpl.colors.Normalize(vmin=c_.min(), vmax=c_.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
    cmap.set_array([])

    color = cmap.to_rgba(2.0 + c_idx)
    print(color)

    ax.errorbar(step_sizes, t1_list_tsppd_rs, yerr = t1_se_tsppd_rs, fmt = ".-", label = "c = {}".format(c), color =color)
    ax.legend()
    recenter = 4*width
    #------Epsilon TS
    n_lines = 7
    c_ = np.arange(1, n_lines + 1)

    norm = mpl.colors.Normalize(vmin=c_.min(), vmax=c_.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Greens)
    cmap.set_array([])

    color = cmap.to_rgba(2.0 + c_idx)
    print(color)

    ax.errorbar(step_sizes, t1_list_ets, yerr = t1_se_ets, fmt = ".-", label = "epsilon = {}".format(epsilon), color =color)
#    ax.legend()

    #-----TS------
    if c_idx == 0:
        ax.errorbar(step_sizes, t1_list_ts, yerr = t1_se_ts, fmt = ".-", label = "TS Wald", color = 'red')
        ax.legend()
    recenter = 4*width
#    ipdb.set_trace()
    #print(t1_se_unif)
#    p1 = ax.bar(ind, t1_list_eg0pt1, width = width, yerr = t1_eg0pt1_se, \
 #               ecolor='black', capsize=capsize, color = 'yellow', edgecolor='black')
    
 #   p3 = ax.bar(ind+width, t1_list_eg0pt3, width = width, yerr = t1_eg0pt3_se, \
 #               ecolor='black', capsize=capsize, color = 'grey', edgecolor='black')
#    step_sizes = [int(np.ceil(n/2)), int(n), int(np.ceil(2*n)), int(np.ceil(4*n))]
#    h = stats.t.ppf(1-0.025, num_sims)*np.sqrt(phi_list*(1-phi_list)/num_sims) #95 CI for Proportion

    
#    p3 = ax.bar(ind + 6*width +2*width - recenter, t1_list_tsppd_rs, width = width, yerr = t1_se_tsppd_rs,     
#           ecolor='black', capsize=capsize, color = 'yellow', edgecolor='black') 
#
#    p4 = ax.bar(ind+2*width - recenter, t1_list_ts, width = width, yerr = t1_se_ts,     
#               ecolor='black', capsize=capsize, color = 'blue', edgecolor='black') 
#
#    p5 = ax.bar(ind+5*width + c_idx*width - recenter, t1_list_tsppd, width = width, yerr = t1_se_tsppd,     
#               ecolor='black', capsize=capsize, color = 'purple', edgecolor='black') 
#
#    if c_idx !=0:
#        p6 = ax.bar(ind+3*width + (c_idx-1)*width - recenter, t1_list_ets, width = width, yerr = t1_se_ets,     
#               ecolor='black', capsize=capsize, color = 'brown', edgecolor='black') 
#
#  
#  #  p2 = ax.bar(ind-width, t1_list_unif, width = width,\
#  #               yerr = t1_se_unif, ecolor='black', \
#  #              capsize=capsize, color = 'red', \
#   #             edgecolor='black')
#
#    if ax_idx == 2 and c_idx != 0:
#      # leg1 = ax.legend((p2[0], p1[0], p3[0], p4[0], p6[0], p5[0]), ("Uniform Wald", 'Epsilon Greedy 0.1 Wald', "Epsilon Greedy 0.3 Wald", "Thompson Sampling Wald", "Epsilon 0.1 Thompson Sampling Wald", "PostDiff Thompson Sampling Wald \n c left to right [0.08, 0.1, 0.12]"), bbox_to_anchor=(1.0, 1.76))  
#       leg1 = ax.legend((p4[0], p6[0], p5[0], p3[0]), ( "Thompson Sampling Wald", "Epsilon Thompson Sampling Wald \n Epsilon left to right [0.1, 0.3]", "PostDiff Thompson Sampling Wald \n c left to right [0.08, 0.1, 0.12]", "PostDiff Thompson Sampling Wald 0.1 \n Resample"), bbox_to_anchor=(1.0, 1.76))  
#    #leg2 = ax.legend(loc = 2)
#    
#       ax.add_artist(leg1)
# #   plt.tight_layout()
#   # plt.title(title)
##    if ax_idx == 6 or ax_idx == 7 or ax_idx == 8:
#    ax.set_xlabel("number of participants = \n n/2, n, 2*n, 4*n")
    ax.set_ylim(0, ymax)
##    ax.set_ylim(0, 0.58)
#    ax.axhline(y=hline, linestyle='--')


    return [t1_list_ts, t1_list_ets, t1_list_tsppd_rs, t1_list_unif] #returns [UR, ppd, TS, ets], in this case, need to return for each step size, but only plotting for one bs, so save step size by model (4x2)


def parse_dir(root, root_cutoffs, num_sims, metric = "Power&T1"):
    arm_prob= 0.5
    arm_prob_list = [0.2, 0.5, 0.8]
    es_list = [0.0, 0.0, 0.0, 0.0]
    es_list = [0.0, 0.0]
   # es_list = [0.3, 0.1]
    n_list = [32, 88, 197, 785]
    n_list = [197, 785]
   # n_list = [32, 88, 785]
    #n_list = [88, 785]
    ymax = 0.28
    root_dir = root + "/num_sims={}armProb={}".format(5000, arm_prob)
    fig_p, ax_p = plt.subplots(1,4, figsize = (12,5))#for phi
    ax_p = ax_p.ravel()
    fig_www, ax_www = plt.subplots()#for phi
    fig, ax = plt.subplots(1,4, figsize = (12,5))
    ax = ax.ravel()
   # ipdb.set_trace()
    num_sims_secb = 5000
    root_ts = "../../../../banditalgorithms/src/RL4RLSectionB/simulation_saves/NoEffect_fixedbs_RL4RLMay8/num_sims={}armProb=0.5".format(num_sims_secb)
#    root_tsppd_rs = "../simulation_saves/TSPPDNoEffectResample/num_sims={}armProb=0.5".format(num_sims)
    root_tsppd_rs = "../simulation_saves/TSPPDNoEffectResampleFast/num_sims={}armProb=0.5".format(num_sims)
    
    root_ppdg = "../simulation_saves/PPDGreedyNoEffectResampleFast/num_sims={}armProb=0.5".format(num_sims) #This is fast, naming not consistent
    root_ets = "../simulation_saves/EpsilonTSNoEffect/num_sims={}armProb=0.5".format(num_sims) #This is fast, naming not consistent

    root_eg_old = "../../../../banditalgorithms/src/RL4RLSectionB/simulation_saves/EpsilonGreedyNoEffect/num_sims={}armProb=0.5".format(num_sims_secb)
    root_unif = "../../../../banditalgorithms/src/RL4RLSectionB/simulation_saves/UniformNoEffect/num_sims={}armProb=0.5".format(num_sims_secb)
    c_list = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3]
    c_list = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3]
    epsilon_list = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.6]
    root_eg = "../simulation_saves/EpsilonGreedyNoEffectFast/num_sims={}armProb=0.5/".format(num_sims) #This is fast, naming not consistent

    c_list = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3]
    c_list_labels = ["c =eps=0.025", "c=eps=0.05", "c=eps=0.075", "c=eps=0.1", "c=eps=0.125", "c=eps=0.15", "c=eps=0.2", "c=0.3/eps=0.6"]
    #epsilon_list = [0.3, 0.3, 0.3, 0.50, 0.525, 0.55, 0.575, 0.6]
    c_idx = 0
    #epsilon_list =c_list 

    table_3_dict = {}
    table_3_dict_old = {}

   # c_list = [0.08, 0.1, 0.12]
    j = 0
    for c in c_list:
        j +=1
        i = 0
        epsilon = epsilon_list[j-1]
        print("EPSILON:", epsilon)
#        epsilon = c

        for n in n_list:
            bs = 1
            n_eg = n
            if n == 197: #haven't run eg for es 0.2 n 197
                n_eg = 32
            es_dir_0pt1 = root_eg_old + "/N={}epsilon={}/".format(n_eg, 0.1)
            es_dir_0pt3 = root_eg_old + "/N={}epsilon={}/".format(n_eg, 0.3)
            ts_dir = root_ts + "/N={}/".format(n)
            ts_dir = root_tsppd_rs + "/N={}c=0.0/".format(n)
            tsppd_dir = root_dir + "/N={}c={}/".format(785, 0.1)
            tsppd_dir_rs = root_tsppd_rs + "/N={}c={}/".format(n, c)

            if c <= 0.1:
                ppdg_dir = root_ppdg + "/N={}c={}/".format(n, 0.1)
                eg_dir = root_eg + "/N={}epsilon={}/".format(n, 0.1)
            else:
                ppdg_dir = root_ppdg + "/N={}c={}/".format(n, 0.2)
                eg_dir = root_eg + "/N={}epsilon={}/".format(n,0.6)

            ets_dir = root_ets + "/N={}epsilon={}/".format(n, epsilon)

            unif_dir = root_ets + "/N={}epsilon={}/".format(n, 0.025)
            unif_dir = root_tsppd_rs + "/N={}c={}/".format(n, 1.0)

#FROM HERE, make list of model dirs, pass into proccess_model_dir_list
            table_3_key_UR = TABLE_3_KEY_UR
            table_3_key_TS = TABLE_3_KEY_TS
            table_3_key_TSPPD = TABLE_3_KEY_TSPPD
            table_3_key_ETS = TABLE_3_KEY_ETS
            table_3_key_PPDG = TABLE_3_KEY_PPDG
            table_3_key_EG = TABLE_3_KEY_EG

            model_dir_list = [ts_dir, tsppd_dir_rs, ets_dir, unif_dir, ppdg_dir, eg_dir]
            model_key_list = [table_3_key_TS, table_3_key_TSPPD, table_3_key_ETS, table_3_key_UR, table_3_key_PPDG, table_3_key_EG]

            process_model_dir_list(model_dir_list = model_dir_list, model_key_list = model_key_list, c=c, n= n, table_3_dict = table_3_dict, metric = metric)
            #Old code from here
#            ipdb.set_trace()
            to_check_ppdg = glob.glob(ppdg_dir + "/*Prior*{}*{}Df*.pkl".format(bs,n))[0] #Has eg, 34 in 348!!
            assert(len(glob.glob(ppdg_dir + "/*Prior*{}*{}Df*.pkl".format(bs,n))) == 1)

            to_check_eg0pt1 = glob.glob(es_dir_0pt1 + "/*Prior*{}*{}Df.pkl".format(bs,n_eg))[0] #Has eg, 34 in 348!!
            assert(len(glob.glob(es_dir_0pt1 + "/*Prior*{}*{}Df.pkl".format(bs,n_eg))) == 1)

            to_check_eg0pt3 = glob.glob(es_dir_0pt3 + "/*Prior*{}*{}Df.pkl".format(bs,n_eg))[0] #Has eg, 34 in 348!!
            assert(len(glob.glob(es_dir_0pt3 + "/*Prior*{}*{}Df.pkl".format(bs,n_eg))) == 1)
          
            #to_check_unif = glob.glob(unif_dir + "/*Uniform*{}*{}Df*.pkl".format(bs, n))[0]
            to_check_unif = glob.glob(unif_dir + "/*Prior*{}*{}Df*.pkl".format(bs, n))[0]
            assert(len(glob.glob(unif_dir + "/*Prior*{}*{}Df*.pkl".format(bs, n))) == 1)

            to_check_ts = glob.glob(ts_dir + "/*Prior*{}*{}Df*.pkl".format(bs,n))[0] #Has eg, 34 in 348!!
            assert(len(glob.glob(ts_dir + "/*Prior*{}*{}Df*.pkl".format(bs,n))) == 1)

            to_check_tsppd = glob.glob(tsppd_dir + "/*Prior*{}*{}Df.pkl".format(bs,785))[0] #Has eg, 34 in 348!!
            assert(len(glob.glob(tsppd_dir + "/*Prior*{}*{}Df.pkl".format(bs,785))) == 1)

            to_check_tsppd_rs = glob.glob(tsppd_dir_rs + "/*Prior*{}*{}Df*.pkl".format(bs,n))[0] #Has eg, 34 in 348!!
            assert(len(glob.glob(tsppd_dir_rs + "/*Prior*{}*{}Df*.pkl".format(bs,n))) == 1)

            print(ets_dir)
            to_check_ets = glob.glob(ets_dir + "/*Prior*{}*{}Df*.pkl".format(bs,n))[0] #Has eg, 34 in 348!! 
#            print(ets_dir)
            assert(len(glob.glob(ets_dir + "/*Prior*{}*{}Df*.pkl".format(bs,n))) == 1)


          #------hists, tables etc
            with open(to_check_eg0pt1, 'rb') as f:
                df_eg0pt1 = pickle.load(f)
            with open(to_check_eg0pt3, 'rb') as f:
                df_eg0pt3 = pickle.load(f)
                                                   
            with open(to_check_unif, 'rb') as f:
                df_unif = pickle.load(f)
            if to_check_ts != None:
                with open(to_check_ts, 'rb') as t:
                    df_ts = pickle.load(t)
            with open(to_check_tsppd, 'rb') as f:
                df_tsppd = pickle.load(f)
            with open(to_check_tsppd_rs, 'rb') as f:
                df_tsppd_rs = pickle.load(f)


            with open(to_check_ets, 'rb') as f:
                df_ets = pickle.load(f)
            with open(to_check_ppdg, 'rb') as f:
                df_ppdg = pickle.load(f)
              
#            ipdb.set_trace()
            rect_key = "TS"
            rect_key = "Drop NA"
            rectify_vars_noNa(df_eg0pt1, alg_key = rect_key)
            rectify_vars_noNa(df_eg0pt3, alg_key = rect_key)
            rectify_vars_noNa(df_ts, alg_key = rect_key)
            rectify_vars_noNa(df_unif, alg_key = rect_key)
       
            assert np.sum(df_eg0pt1["wald_type_stat"].isna()) == 0
            assert np.sum(df_eg0pt1["wald_pval"].isna()) == 0
     
#            next_df = line_plot(df_eg0pt1 = df_eg0pt1, df_eg0pt3 = df_eg0pt3,\
#                                         df_unif = df_unif, df_ts = df_ts, df_tsppd = df_tsppd, df_ets = df_ets,\
#                                                 n = n, num_sims = num_sims,
#                                                   ax = ax[i], ax_idx = i, epsilon = epsilon, c=c, ymax = 0.35, hline = 0.05,df_tsppd_rs = df_tsppd_rs, c_idx = c_idx)
#    #--------------------------------------------------
            next_df = stacked_bar_plot_with_cutoff(df_eg0pt1 = df_eg0pt1, df_eg0pt3 = df_eg0pt3,\
                                         df_unif = df_unif, df_ts = df_ts, df_tsppd = df_tsppd, df_ets = df_ets, c_list = c_list, df_ppdg = df_ppdg, \
                                                 n = n, num_sims = num_sims,
                                                   ax = ax[i], ax_idx = i, epsilon = epsilon, c=c, ymax = ymax, hline = 0.05,df_tsppd_rs = df_tsppd_rs, c_idx = c_idx, table_3_dict = table_3_dict_old)
    #--------------------------------------------------




            title_cond1 = "Proportion of Samples in Condition 1 For n = {} \n Across {} Simulations \n No Effect $p_1$ = $p_2$ = 0.5".format(n, num_sims)

            if c == 0.1:
                hist_cond1(df_eg0pt1 = df_eg0pt1, df_eg0pt3 = df_eg0pt3,\
                     df_unif = df_unif, df_ts = df_ts, df_tsppd = df_tsppd, df_ets = df_ets,\
                       n = n, num_sims = num_sims,\
                             title = title_cond1)
     
          #  ipdb.set_trace()

            title_scatter_ratio = "Minimum Sample Size Ratio \n Min($\\frac{n_1}{n_2}$, $\\frac{n_2}{n_1}$)" + " For n = {} \n Across {} Simulations \n No Effect $p_1$ = $p_2$ = 0.5".format(n, num_sims)

          #  scatter_ratio(to_check_eg0pt1 = to_check_eg0pt1, to_check_eg0pt3 = to_check_eg0pt3,\
           #                              to_check_unif = to_check_unif, to_check_ts = to_check_ts,\
           #                                    title = title_scatter_ratio, \
           #                                    n = n, num_sims = num_sims)

            title_table = "TODO"
    #        table_means_diff(df_ts = df_ts, df_eg0pt1 = df_eg0pt1, df_eg0pt3 = df_eg0pt3, df_unif = df_unif, n = n, num_sims = num_sims, \
     #                       title = title_table)

    #        scatter_correlation_helper_outer(df_eg0pt1 = df_eg0pt1, df_eg0pt3 = df_eg0pt3,\
    #                                     df_unif = df_unif, df_ts = df_ts,\
    #                                           n = n, num_sims = num_sims)  #Title created in helper fn


        #    title_pval = "Chi Squared P value Disutrbtuion For n = {} \n Across {} Simulations".format(n, num_sims)
            
            
         #   hist_pval(to_check = to_check_eg0pt1, to_check_unif = to_check_unif, to_check_ts = to_check_ts, n = n, num_sims = num_sims, load_df = True, \
          #               title = title_pval, plot = True)

            title_mean1 = "Mean 1 ($\hatp_1$ with MLE) Disutrbtuion For n = {} \n Across {} Simulations \n No Effect $p_1$ = $p_2$ = 0.5".format(n, num_sims)

            title_mean2 = "Mean 2 ($\hatp_2$ with MLE) Disutrbtuion For n = {} \n Across {} Simulations \n No Effect $p_1$ = $p_2$ = 0.5".format(n, num_sims)

    #        hist_means_bias(to_check_eg0pt1 = to_check_eg0pt1, to_check_eg0pt3 = to_check_eg0pt3,\
    #                                     to_check_unif = to_check_unif, to_check_ts = to_check_ts,\
    #                                           title = title_mean1, \
    #                                           n = n, num_sims = num_sims, mean_key = "mean_1")
    #
    #        hist_means_bias(to_check_eg0pt1 = to_check_eg0pt1, to_check_eg0pt3 = to_check_eg0pt3,\
    #                                     to_check_unif = to_check_unif, to_check_ts = to_check_ts,\
    #                                           title = title_mean2, \
    #                                           n = n, num_sims = num_sims, mean_key = "mean_2")

            title_diff = "Difference in Means (|$\hatp_1$ - $\hatp_2$| with MLE) Disutrbtuion For n = {} \n Across {} Simulations \n No Effect $p_1$ = $p_2$ = 0.5".format(n, num_sims)

    #        hist_means_diff(df_eg0pt1 = df_eg0pt1, df_eg0pt3 = df_eg0pt3,\
    #                                     df_unif = df_unif, df_ts = df_ts,\
    #                                           title = title_diff, \
    #                                           n = n, num_sims = num_sims)

            title_imba = "Sample Size Imbalance (|$\\frac{n_1}{n} - 0.5$|"+" Disutrbtuion For n = {} \n Across {} Simulations \n No Effect $p_1$ = $p_2$ = 0.5".format(n, num_sims)

    #        hist_imba(df_eg0pt1 = df_eg0pt1, df_eg0pt3 = df_eg0pt3,\
    #                                     df_unif = df_unif, df_ts = df_ts,\
    #                                           title = title_imba, \
    #                                           n = n, num_sims = num_sims)

    #        title_wald = "Wald Statistic Sampling Disutrbtuion For n = {} \n Across {} Simulations \n No Effect $p_1$ = $p_2$ = 0.5".format(n, num_sims)

     #       hist_wald(to_check_eg0pt1 = to_check_eg0pt1, to_check_eg0pt3 = to_check_eg0pt3,\
     #                                    to_check_unif = to_check_unif, to_check_ts = to_check_ts,\
     #                                          title = title_wald, \
     #                                          n = n, num_sims = num_sims)

     #       title_kde = "Wald Statistic KDE Sampling Disutrbtuion For n = {} \n Across {} Simulations \n No Effect $p_1$ = $p_2$ = 0.5".format(n, num_sims)#

    #        KDE_wald(to_check_eg0pt1 = to_check_eg0pt1, to_check_eg0pt3 = to_check_eg0pt3,\
     #                                    to_check_unif = to_check_unif, to_check_ts = to_check_ts,\
      #                                         title = title_kde, \
       #                                        n = n, num_sims = num_sims)

            title_ap1 = "Arm 1 Assignment Probability Disutrbtuion For n = {} \n Across {} Simulations \n No Effect $p_1$ = $p_2$ = 0.5".format(n, num_sims)
            title_ap2 = "Arm 2 Assignment Probability Disutrbtuion For n = {} \n Across {} Simulations \n No Effect $p_1$ = $p_2$ = 0.5".format(n, num_sims)

            actions_dir_ts = ts_dir + "bbEqualMeansEqualPriorburn_in_size-batch_size={}-{}".format(bs, bs)
                                                  
    #        probs_dict = calculate_assgn_prob_by_step_size(actions_root = actions_dir_ts, num_samples=1000, num_actions = 2, cached_probs={}, 
    #                  prior = [1,1], binary_rewards = True, \
    #                  config = {}, n = n,\
    #                  num_sims = num_sims, batch_size = 1, no_effect = True)
    #
    #
    #
    #        hist_assignprob(to_check_eg0pt1 = to_check_eg0pt1, to_check_eg0pt3 = to_check_eg0pt3,\
    #                                     to_check_unif = to_check_unif, to_check_ts = to_check_ts, ts_ap_df = probs_dict,\
    #                                           title = title_ap1, \
    #                                           n = n, num_sims = num_sims, mean_key_of_interest = "mean_1", mean_key_other = "mean_2")
    #
    #
    #        hist_assignprob(to_check_eg0pt1 = to_check_eg0pt1, to_check_eg0pt3 = to_check_eg0pt3,\
    #                                     to_check_unif = to_check_unif, to_check_ts = to_check_ts, ts_ap_df = probs_dict,\
    #                                           title = title_ap2, \
    #                                           n = n, num_sims = num_sims, mean_key_of_interest = "mean_2", mean_key_other = "mean_1")
    #

    #--------------------------------------------------
            ax[i].set_title("n = {}".format(n))
#            ipdb.set_trace()
            plot_phi(df_tsppd_rs, num_sims, n, c, ax_p[i]) #averaged over sims
            if n == 32:
                ax[i].set_ylabel("False Positive Rate")
            if n == 785 and c != 0.3:
                plot_phi_www(df_tsppd_rs, num_sims, n, c, ax_www) #averaged over sims
            i += 1

            df = pd.DataFrame(next_df, columns = ["n/2","n","2n","4n"])
            df.index = ["Thompson Sampling Wald", "Epsilon Thompson Sampling Wald","Thompson Sampling PostDiff Wald", "Uniform Random Wald"]
            #[UR, ppd, TS, ets]
            save_dir = "../simulation_analysis_saves/Tables/FPR/c_and_epsilon={}/N={}/".format(c, n)

            Path(save_dir).mkdir(parents=True, exist_ok=True)


            df.to_csv(save_dir + "/FPR_n={}_numsims={}.csv".format(n, num_sims)) 
            #---phi

        c_idx +=1

           
    fig_p.tight_layout(rect=[0, 0.03, 1, 0.89])
    fig_www.tight_layout(rect=[0, 0.03, 1, 0.83])
    save_dir = "../simulation_analysis_saves/num_sims={}/phi_plots/NoEffect/".format(num_sims)
    save_dir_www = "../simulation_analysis_saves/num_sims={}/phi_plots_WWW/NoEffect/".format(num_sims)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(save_dir_www).mkdir(parents=True, exist_ok=True)

    title = "$\hat \phi$ Across {} Simulations $p_1^* = p_2^* = 0.5$ \n $\phi$ := p($|p_1 - p_2| < c$)".format(num_sims)
    title_www = "$\hat \phi$ Across {} Simulations $p_1^* = p_2^* = 0.5$ \n $\phi$ := p($|p_1 - p_2| < c$)".format(num_sims)
    
    fig_p.suptitle(title)
    fig_www.suptitle(title_www)
    fig_www.savefig(save_dir_www+ "/" + title_www +".png")

    if metric == "Power&T1":
        title = "Type 1 Error Rate \n Across {} Simulations".format(num_sims)
        title = "False Positive Rate \n Across {} Simulations".format(num_sims)
            #ax[i].set_title(title, fontsize = 55)
            #i +=1
            #fig.suptitle("Type One Error Rates Across {} Simulations".format(num_sims))
    else:
        title = "Reward"
    fig.suptitle(title)
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    #handles, labels = ax[i-1].get_legend_handles_labels()
    
    #fig.legend(handles, labels, loc='upper right', prop={'size': 50})
        #fig.tight_layout()
    save_dir = "../simulation_analysis_saves/power_t1_plots"
    if not os.path.isdir(save_dir):
          os.mkdir(save_dir)
#    print("saving to ", "plots/{}.png".format(title))
    
    #fig.set_tight_layout(True)
    fig.tight_layout()
    fig.subplots_adjust(top=.8)
    

#    ipdb.set_trace()
    fig.savefig(save_dir + "/{}.svg".format(title), bbox_inches = 'tight')
   # plt.show()
    fig.clf()
    plt.close(fig)
  #  convert_table3_dict_totable(table_3_dict, c_list = c_list, n_list = n_list, num_sims = num_sims, es_list = es_list, metric = metric, c_list_labels = c_list_labels)
    convert_table3_dict_totable_loop(table_3_dict, c_list = c_list, n_list = n_list, num_sims = num_sims, es_list = es_list, metric = metric, c_list_labels = c_list_labels, keys_list = model_key_list)
#    convert_table3_dict_totable(table_3_dict, c_list = c_list, n_list = n_list, num_sims = num_sims, es_list = es_list, metric = metric, c_list_labels = c_list_labels)



        
if __name__ == "__main__":
    root = "../simulation_saves/TSPPDNoEffect_c=0pt1"
    root = "../simulation_saves/TSPPDNoEffect"
    #parse_dir(root, root_cutoffs)
    num_sims = 500
    num_sims = 5000
    num_sims = 10000
    parse_dir(root, root, num_sims)
    parse_dir(root, root, num_sims, metric = "PropMaj")
    parse_dir(root, root, num_sims, metric = "PropEps")


