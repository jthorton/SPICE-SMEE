import deepchem as dc
import datasets
import numpy as np

# this uses mapped hydrogen smiles is that an issue?
# load the dataset and get the smiles
dataset = datasets.load_from_disk('mace_filtered_spice_114')
smiles = []
for entry in dataset:
    smiles.append(entry['smiles'])

print(len(smiles))

Xs = np.zeros(len(smiles))
# make the dc dataset from the smiles
dc_dataset = dc.data.DiskDataset.from_numpy(X=Xs, ids=smiles)

maxminspliter = dc.splits.MaxMinSplitter()
# split and save
train_dataset, test_dataset = maxminspliter.train_test_split(dataset=dc_dataset, frac_train=0.95, train_dir='maxmin-train', test_dir='maxmin-test')
