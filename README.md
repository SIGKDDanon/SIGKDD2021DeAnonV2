# Requirements
pandas==1.1.5
ipdb==0.13.2
scipy==1.4.1
numpy==1.18.1
statsmodels==0.11.0
matplotlib==3.1.2

# Running Simulations
In the folder PostDiffMixture/simulations_folder/simulation_scripts we have provided shell scripts to run simulations. See the contained README.txt (PostDiffMixture/simulations_folder/simulation_scripts/README.txt) for a description of each shell script. 
# Analyzing Simulations
After simulations have been run, they will be saved to the folder PostDiffMixture/simulations_folder/simulation_saves. You can then analyze these simulations using the python scripts in the folder /PostDiffMixture/simulations_folder/simulation_analysis_scripts. See the contained README.txt (PostDiffMixture/simulations_folder/simulation_analysis_scripts/README.txt) for a description of each file. Note that the analysis files assume you have run and saved PostDiff TS for c in {XX}, Epsilon TS for epsilon in {XX}, PostDiff Greedy for c in {XX}, and Epsilon Greedy for epsilon in {XX}. If you have not run these then you will need to modify the analysis code to remove reference to these saves.
