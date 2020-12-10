# Gamma-minimax-learninng

Simulation code for "Leveraging vague prior information in general models via iteratively constructed Gamma-minimax estimators".

Folders with name starting with "univariate_mean" correspond to the simulation on estimating the mean. `univariate_mean/` corresponds to the space of all linear decisions and SGDmax. `univariate_mean_fictitious_play/` corresponds to the space of all linear decisions and fictitious play. `univariate_mean_skn` and `univariate_mean_nn` correspond to statistical knowledge networks and more naive neural networks learned with SGDmax, respectively.

`n_new_categories/` corresponds to the simulation on predicting the expected number of new categories. `entropy/` corresponds to the simulation on estimating the entropy. Among the subfolders, `1/`, `2/` and `3/` correspond to strongly informative, weakly informative and almost non-informative prior knowledge.
