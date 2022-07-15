# Molecular Representation Learning via Heterogeneous Motif Graph Neural Networks
This is an official PyTorch implementation of the experiments in the following paper:\
[<b>Zhaoning Yu</b>, Hongyang Gao. Molecular Representation Learning via Heterogeneous Motif Graph Neural Networks. ICML 2022](https://proceedings.mlr.press/v162/yu22a.html)

## Requirements

## Part 1: Heterogeneous Motif Graph Construction
You can use preprocess.py to construct HM-graph for TUDataset. Change the parameter of drop_node() function in the ops.py to drop noises in the motif dictionary.\
You can use preprocess_hiv.py and preprocess_pcba.py to construct HM-graph for ogbg-molhiv and ogbg-pcba dataset. For ogbg-pcba dataset, because there are 11 graphs do not have motifs, you need to substract 11 from self.num_cliques.

## Part 2: Reproduce the results
Run main.py for TUDataset.\
Run main_ogbg_molhiv.py for ogbg-molhiv.\
Run main_molpcba.py for ogbg-pcba.
