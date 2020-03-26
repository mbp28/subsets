bsub -R "rusage[mem=10000, ngpus_excl_p=1]" "python -m subsets.L2X.imdb_word.validate_batch"
