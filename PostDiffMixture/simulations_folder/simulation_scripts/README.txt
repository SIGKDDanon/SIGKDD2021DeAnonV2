RunEffectSizeSims: TS with Unif, with effect, with batch size as percentage of sample size
RunEffectSizeSimsSameArms: TS with Unif, no effect, with batch size as percentage of sample size
RunEffectSizeSimsTSPPD: PostDiff TS with non zero effect size
RunEffectSizeSimsSameArmsTSPPD: PostDiff TS with no effect
RunEffectSizeSimsPPDGreedy: PostDiff Greedy with effect
RunEffectSizeSimsSameArmsPPDGreedy: PostDiff Greedy with no effect
RunEffectSizeSimsEpsilonGreedy: EG, with effect
RunEffectSizeSimsEpsilonTS: Epsilon TS non zero effect size
RunEffectSizeSimsSameSizeEpsilonTS: Epsilon TS with no effect
RunEffectSizeSimsSameArmsEpsilonGreedy: EG, no effect
RunEffectSizeSims_fixedbs: TS with Unif, with effect, with batch size a fixed sample size
RunEffectSizeSimsSameArms_fixedbs: TS with Unif, no effect, with batch size as a fixed sample size

The files run_effect_sizes_simulations_XX.py are called by the above shell scripts to run simulations. Note than those containing the word "fast" will use the updated method for running bandit trials, whereas those without are using the previous method which is much slower, and will save a full trajectory of rewards. Double check the shell script you run to see whether it is running a fast version or not.

