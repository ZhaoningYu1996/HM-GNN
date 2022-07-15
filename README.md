# Molecular Representation Learning via Heterogeneous Motif Graph Neural Networks
This is an official PyTorch implementation of the experiments in the following paper:\
[Molecular Representation Learning via Heterogeneous Motif Graph Neural Networks](https://proceedings.mlr.press/v162/yu22a.html) (ICML 2022)\
<b>Zhaoning Yu</b>, Hongyang Gao

## Requirements
```
pytorch                       1.9.0
rdkit-pypi                    2021.9.2
ogb                           1.3.1
dgl                           0.6.1
networkx
```
## Part 1: Heterogeneous Motif Graph Construction
Run ```python preprocess.py``` to construct HM-graph for TUDataset.\
Change the parameter of drop_node() function in the ops.py to drop noises in the motif dictionary.\
Run ```python preprocess_hiv.py``` and ```python preprocess_pcba.py``` to construct HM-graph for ogbg-molhiv and ogbg-pcba dataset.\
For ogbg-pcba dataset, because there are 11 graphs do not have motifs, you need to substract 11 from self.num_cliques.

## Part 2: Training and evaluation
Run ```python main.py``` for TUDataset.\
Run ```python main_ogbg_molhiv.py``` for ogbg-molhiv.\
Run ```python main_molpcba.py``` for ogbg-pcba.

## Cite
If you find this repo or paper to be useful, please cite our paper.
```
@inproceedings{yu2022molecular,
  title={Molecular Representation Learning via Heterogeneous Motif Graph Neural Networks},
  author={Yu, Zhaoning and Gao, Hongyang},
  booktitle={International Conference on Machine Learning},
  pages={25581--25594},
  year={2022},
  organization={PMLR}
}
```
