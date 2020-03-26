for seed in {1..20}
do
for tau in 0.1 0.5 1.0 2.0 5.0
do
for task in l2x subsets knapsack lml
do
for k in 2 4 6 8 10
do
#bsub -R "rusage[mem=10000]" "python -m subsets.L2X.imdb_word.explain --train --k $k --task $task --tau $tau --seed $seed"
bsub -R "rusage[mem=10000, ngpus_excl_p=1]" "python -m subsets.L2X.imdb_word.explain --train --k $k --task $task --tau $tau --seed $seed"
done
done
done
done
