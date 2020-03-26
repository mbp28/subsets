for task in l2x subsets
do
for tau in 0.1 0.5 1.0 2.0 5.0
do
python -m subsets.L2X.imdb_word.validate_explanation --task ${task} --tau ${tau}
done
done
