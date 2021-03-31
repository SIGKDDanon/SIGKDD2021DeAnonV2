# Requirements
pandas==1.1.5 \
ipdb==0.13.2 \
scipy==1.4.1 \
numpy==1.18.1 \
statsmodels==0.11.0 \
matplotlib==3.1.2

# Running Simulations
In the folder PostDiffMixture/simulations_folder/simulation_scripts we have provided shell scripts to run simulations. For a description of each shell script, see the contained README.txt:

PostDiffMixture/simulations_folder/simulation_scripts/README.txt

# Analyzing Simulations
After simulations have been run, they will be saved to the folder 

PostDiffMixture/simulations_folder/simulation_saves

You can then analyze these simulations using the python scripts in the folder 

PostDiffMixture/simulations_folder/simulation_analysis_scripts. 

For a description of each file, see the contained README.txt:

PostDiffMixture/simulations_folder/simulation_analysis_scripts/README.txt

Note that the analysis files assume you have run and saved PostDiff TS for c in {0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 1.0}, Epsilon TS for epsilon in {0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.6}, PostDiff Greedy for c in {0.1, 0.2}, and Epsilon Greedy for epsilon in {0.1, 0.6}. If you have not run these then you will need to modify the analysis code to remove reference to these saves.
