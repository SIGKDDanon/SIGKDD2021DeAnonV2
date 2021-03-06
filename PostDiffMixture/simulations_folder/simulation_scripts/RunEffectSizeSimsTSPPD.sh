#------NEW
numSims=2
numSims=5000
numSims=10000
#variancesT=(5.0 2.0 1.25); # Note: we're not crossing nsT and variancesT because variancesT is actual variance with ES with that num steps

timestamp=$(date +%F_%T)

simSetDescriptive="../simulation_saves/TSPPDIsEffectTEST" #Give descriptive name for directory to save set of sims
simSetDescriptive="../simulation_saves/TSPPDIsEffectTest" #Give descriptive name for directory to save set of sims
simSetDescriptive="../simulation_saves/TSPPDIsEffect" #Give descriptive name for directory to save set of sims
simSetDescriptive="../simulation_saves/TSPPDIsEffectResample" #Give descriptive name for directory to save set of sims
simSetDescriptive="../simulation_saves/TSPPDIsEffectResampleFast" #Give descriptive name for directory to save set of sims
mkdir -p $simSetDescriptive

effectSizesB=(0.1 0.2 0.3 0.5);
effectSizesB=(0.1 0.2);

#effectSizesB=(0.3);

#nsT=(394 64 26);
#nsB=(785 88 32);

arrayLength=${#effectSizesB[@]};
#nsB=(32); #hard coded sample sizes corresponding to es

bsProps=(0.25);
bsProps_len=${#bsProps[@]};
armProbs=(0.5); #as default
armProbs_len=${#armProbs[@]};

#for ((a=0; a<$armProbs_len; a++)); do
armProb=${armProbs[0]}
root_armProb=$simSetDescriptive/"num_sims="$numSims"armProb="$armProb
echo $armProb

#c=0.1
#c_list=(0.025 0.05 0.08 0.1 0.12 0.2 0.3); #[0.025, 0.05, 0.1, 0.2, 0.3],  [0.08, 0.1, 0.12] 
#c_list=(0.08 0.1 0.12); #[0.025, 0.05, 0.1, 0.2, 0.3],  [0.08, 0.1, 0.12] 
c_list=(0.025 0.05 0.1 0.2 0.3);
c_list=(0.0);
c_list=(0.225 0.25 0.3);
c_list=(0.025 0.05 0.075 0.1 0.125 0.15 0.2);
c_list=(0.0 0.3 1.0);

c_length=${#c_list[@]};
for ((i=0; i<$arrayLength; i++)); do
    curN=${nsB[$i]}
    curEffectSize=${effectSizesB[$i]}
    for ((j=0; j<$c_length; j++)); do
        c=${c_list[$j]}
	root_armProb_es=$root_armProb/"es="$curEffectSize"c="$c
       # root_armProb_es=$root_armProb/"es="$curEffectSize

       # for ((j=0; j<$bsProps_len; j++)); do
	curProp=${bsProps[0]}
	
	batch_size_fl=$(awk -v curN="${curN}" -v curProp="${curProp}" 'BEGIN{print (curN*curProp)}')
	batch_size=${batch_size_fl%.*}
	batch_size=1 #using 1 for now
	echo $batch_size
	burn_in_size=$batch_size
       # mkdir -p $root_dir

	#TS---------------
	root_armProb_es_prop_ts=$root_armProb_es"/bbUnEqualMeansEqualPriorburn_in_size-batch_size="$burn_in_size"-"$batch_size #root for this sim, split by equals for bs
      
	directoryName_ts=$root_armProb_es_prop_ts
	echo $directoryName_ts
	mkdir -p $directoryName_ts
	
	python3 run_effect_size_simulations_beta_fast_TSPPD.py \
	$curEffectSize"-"$armProb $numSims $directoryName_ts "Thompson" $c 2> $directoryName_ts"/errorOutput.log" > $directoryName_ts"/output.log" &
	echo $!
       
	#from equalmeans for reference
	#python3 le_experiments/run_effect_size_simulations_beta.py \
	#$armProb,$armProb $numSims $directoryName_ts "Thompson" "armsEqual" $curN 2> $directoryName_ts"/errorOutput.log" > $directoryName_ts"/output.log" &
	#echo $!

	#Uniform------------
#	root_armProb_es_prop_unif=$root_armProb_es"/bbUnEqualMeansUniformburn_in_size-batch_size="$burn_in_size"-"$batch_size;
#	directoryName_unif=$root_armProb_es_prop_unif
#	mkdir -p $directoryName_unif
    #    python3 run_effect_size_simulations_beta_PPD_TS.py \
    #    $curEffectSize"-"$armProb $numSims $directoryName_unif "uniform" 2> $directoryName_unif"/errorOutput.log" > $directoryName_unif"/output.log" &
	
	#from equalmeans for reference
	#python3 le_experiments/run_effect_size_simulations_beta.py \
	#$armProb,$armProb $numSims $directoryName_unif "uniform" "armsEqual" $curN 2> $directoryName_unif"/errorOutput.log" > $directoryName_unif"/output.log" &
    done
done
#done



