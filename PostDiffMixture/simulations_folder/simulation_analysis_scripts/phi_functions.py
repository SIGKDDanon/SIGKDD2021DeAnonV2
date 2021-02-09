import ipdb as pb
import pandas as pd
import numpy as np
from scipy.stats import beta

CUML_KEY = "prop_exploring_ppd_cuml"  
SNAPSHOT_KKEY = "exploring_ppd_at_this_n"

def plot_phi(df, num_sims, n, c, ax, es = 0):
    """
    get prop in cond 1 for when exploring
    """
#    pb.set_trace()
    step_sizes = [int(np.ceil(n/2)), int(n), int(np.ceil(2*n)), int(np.ceil(4*n))]
    
    extra_steps = [3,4,5]
    phi_init = np.sum(np.abs(beta.rvs(1, 1, size=5000) - beta.rvs(1, 1, size=5000)) < c)/5000
    phi_list = [phi_init] #per step sizes
   
    for num_steps in step_sizes:#loop over step sizes, make a hist for each
        print("num_steps", num_steps)
        
#        df[df["num_steps"] == 16]["exploring_ppd_at_this_n"].mean() 
        df_for_num_steps = df[df['num_steps'] == num_steps].dropna()

        phi = df_for_num_steps[SNAPSHOT_KKEY].mean() 
        phi_list.append(phi)

#    pb.set_trace()
    phi_list = np.array(phi_list)
    se = np.sqrt(phi_list*(1-phi_list))/np.sqrt(num_sims)
   # h = se * stats.t.ppf((1 + 0.95) / 2., num_sims-1)
   # h = stats.t.ppf(1-0.025, num_sims)*np.sqrt(phi_list*(1-phi_list)/num_sims) #95 CI for Proportion
    #print(phi_list, num_sims)
    step_sizes = [0, int(np.ceil(n/2)), int(n), int(np.ceil(2*n)), int(np.ceil(4*n))]
    if c == 0.025:  #0.05
        phi_list_0 = [0, 0, 0, 0, 0]
        phi_list_0 = np.array(phi_list_0)
        ax.errorbar(step_sizes, phi_list_0,yerr = None,fmt = ".-", label = "c = 0")
    ax.errorbar(step_sizes, phi_list,yerr = None,fmt = ".-", label = "c = {}".format(c))
#  else:
  #      ax.errorbar(step_sizes, phi_list,yerr = None,fmt = ".-", label = "c = {}".format(c))
    ind = np.arange(len(step_sizes))
    ind = step_sizes
    ax.set_xticks(ind)
    print(step_sizes)
    ax.set_xticklabels(step_sizes)
    ax.tick_params(axis='x', rotation=45)

    eps_dec = step_sizes
    eps_dec[0] = eps_dec[0] + 1
    print("EPS DEC")
    print(1.0/np.array(eps_dec))
    eps_dec = np.arange(step_sizes[-1])
    ax.errorbar(eps_dec, 1.0/np.sqrt(np.array(eps_dec)), color='black', linestyle='--', linewidth = 0.5)
    #ax.tick_params(axis='x')

#    ax.set_xlabel("number of participants = \n n/2, n, 2*n, 4*n")
    #if es == 0.1: 
    ax.set_xlabel("number of participants")
    ax.set_ylim(-0.05, 1.05)
    if es == 0:
        ax.set_ylabel("$\hat \phi_t$")
    if es != 0:
        handles, labels = ax.get_legend_handles_labels()
        #ax.legend(handles[::-1], labels[::-1], loc='upper left')
        ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.1, 1.03))
       # ax.legend(handles[::-1], labels[::-1])
    #leg2 = ax.legend(loc = 2)
 

def plot_phi_www(df, num_sims, n, c, ax = None, es = 0):
    """
    get prop in cond 1 for when exploring
    """
#    pb.set_trace()
         

    step_sizes = [int(np.ceil(n/2)), int(n), int(np.ceil(2*n)), int(np.ceil(4*n))]
    #step_sizes = [int(np.ceil(n/2)), int(n)]
    
    extra_steps = [3,4,5]
    phi_init = np.sum(np.abs(beta.rvs(1, 1, size=5000) - beta.rvs(1, 1, size=5000)) < c)/5000
    phi_list = [phi_init] #per step sizes
   
  #  if c != 0:
    for num_steps in step_sizes:#loop over step sizes, make a hist for each
        print("num_steps", num_steps)
        
    #    df[df["num_steps"] == 16]["exploring_ppd_at_this_n"].mean() 
        df_for_num_steps = df[df['num_steps'] == num_steps].dropna()

        phi = df_for_num_steps[SNAPSHOT_KKEY].mean() 
        phi_list.append(phi)

    # pb.set_trace()
    phi_list = np.array(phi_list)
    se = np.sqrt(phi_list*(1-phi_list))/np.sqrt(num_sims)
   # h = se * stats.t.ppf((1 + 0.95) / 2., num_sims-1)
   # h = stats.t.ppf(1-0.025, num_sims)*np.sqrt(phi_list*(1-phi_list)/num_sims) #95 CI for Proportion
    #print(phi_list, num_sims)
    step_sizes = [0, int(np.ceil(n/2)), int(n)]
    step_sizes = [0, int(np.ceil(n/2)), int(n), int(np.ceil(2*n)), int(np.ceil(4*n))]
    ax.errorbar(step_sizes, phi_list,yerr = None,fmt = ".-", label = "c = {}".format(c))
#  else:
  #      ax.errorbar(step_sizes, phi_list,yerr = None,fmt = ".-", label = "c = {}".format(c))
    ind = np.arange(len(step_sizes))
    ind = step_sizes
    ax.set_xticks(ind)
    print(step_sizes)
    ax.set_xticklabels(step_sizes)
    #ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='x')

   # ax.set_xlabel("number of participants = \n n/2, n, 2*n, 4*n")
    ax.set_xlabel("number of participants")
    ax.set_ylabel("$\hat \phi$")
    ax.set_ylim(-0.05, 1.05)
    print('HERHEHREHRHERHEHR')
    eps_dec = step_sizes
    eps_dec[0] = eps_dec[0] + 1
    ax.errorbar(step_sizes, 1.0/np.array(eps_dec), color='black', linestyle='--')

    if c == 0.1 or 1:
        print("PHI 0")
        phi_list_0 = [0, 0, 0, 0, 0]
        phi_list_0 = np.array(phi_list_0)
        ax.errorbar(step_sizes, phi_list_0,yerr = None,fmt = ".-", label = "c = 0")
    if n == 785:
        handles, labels = ax.get_legend_handles_labels()
        #ax.legend(handles[::-1], labels[::-1], loc='upper left')
        ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.1, 1.05))
    #leg2 = ax.legend(loc = 2)
           
#        fig, ax = plt.subplots(1,4, figsize = (12,5))#for phi
#        fig.tight_layout(rect=[0, 0.03, 1, 0.87])
#        save_dir = "../simulation_analysis_saves/phi_plots_WWW/NoEffect/"
#        Path(save_dir).mkdir(parents=True, exist_ok=True)
#
#        title = "$\hat \phi$ Across {} Simulations $p_1 = p_2 = 0.5$ \n $\phi$ := p($|p_1 - p_2| < c$) \n = {}".format(num_sims, n)
#        
#        fig.suptitle(title)
#        fig.savefig(save_dir + "/" + title +".png")
 

def plot_phi_www_multi(df, df_ne, num_sims, n, c, ax = None, es = 0):
    """
    get prop in cond 1 for when exploring
    """
#    pb.set_trace()
         

    step_sizes = [int(np.ceil(n/2)), int(n), int(np.ceil(2*n)), int(np.ceil(4*n))]
    #step_sizes = [int(np.ceil(n/2)), int(n)]
    
    extra_steps = [3,4,5]
    phi_init = np.sum(np.abs(beta.rvs(1, 1, size=5000) - beta.rvs(1, 1, size=5000)) < c)/5000
    phi_list = [phi_init] #per step sizes
   
    for num_steps in step_sizes:#loop over step sizes, make a hist for each
        print("num_steps", num_steps)
        
#        df[df["num_steps"] == 16]["exploring_ppd_at_this_n"].mean() 
        df_for_num_steps = df[df['num_steps'] == num_steps].dropna()

        phi = df_for_num_steps[SNAPSHOT_KKEY].mean() 
        phi_list.append(phi)

#    pb.set_trace()
    phi_list = np.array(phi_list)
    se = np.sqrt(phi_list*(1-phi_list))/np.sqrt(num_sims)
   # h = se * stats.t.ppf((1 + 0.95) / 2., num_sims-1)
   # h = stats.t.ppf(1-0.025, num_sims)*np.sqrt(phi_list*(1-phi_list)/num_sims) #95 CI for Proportion
    #print(phi_list, num_sims)
    step_sizes = [0, int(np.ceil(n/2)), int(n)]
    step_sizes = [0, int(np.ceil(n/2)), int(n), int(np.ceil(2*n)), int(np.ceil(4*n))]
    ax.errorbar(step_sizes, phi_list,yerr = None,fmt = ".-", label = "c = {}".format(c))
#  else:
  #      ax.errorbar(step_sizes, phi_list,yerr = None,fmt = ".-", label = "c = {}".format(c))
    ind = np.arange(len(step_sizes))
    ind = step_sizes
    ax.set_xticks(ind)
    print(step_sizes)
    ax.set_xticklabels(step_sizes)
    #ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='x')

   # ax.set_xlabel("number of participants = \n n/2, n, 2*n, 4*n")
    ax.set_xlabel("number of participants")
    ax.set_ylabel("$\hat \phi$")
    #ax.set_ylim(0.0, 1.0)
    ax.set_ylim(-0.05, 1.05)
    if n == 785:
        handles, labels = ax.get_legend_handles_labels()
        #ax.legend(handles[::-1], labels[::-1], loc='upper left')
        ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.1, 1.05))
    #leg2 = ax.legend(loc = 2)
           
#        fig, ax = plt.subplots(1,4, figsize = (12,5))#for phi
#        fig.tight_layout(rect=[0, 0.03, 1, 0.87])
#        save_dir = "../simulation_analysis_saves/phi_plots_WWW/NoEffect/"
#        Path(save_dir).mkdir(parents=True, exist_ok=True)
#
#        title = "$\hat \phi$ Across {} Simulations $p_1 = p_2 = 0.5$ \n $\phi$ := p($|p_1 - p_2| < c$) \n = {}".format(num_sims, n)
#        
#        fig.suptitle(title)
#        fig.savefig(save_dir + "/" + title +".png")
             
