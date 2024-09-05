import deepchem as dc
import datasets

# load spice 
spice_dataset = datasets.load_from_disk('mace_filtered_spice_114')

train_dataset = dc.data.DiskDataset(data_dir='maxmin-train')
test_dataset = dc.data.DiskDataset(data_dir='maxmin-test')
train_index, test_index = [], []

# extract the training ids
for i, entry in enumerate(spice_dataset):
    if entry['smiles'] in train_dataset.ids:
        train_index.append(i)
    elif entry['smiles'] in test_dataset.ids:
        test_index.append(i)
    else:
        raise RuntimeError('The smiles was not in traing or testing')
    
print(len(train_index), len(test_index), len(spice_dataset))
# split the dataset and save it
train_split = spice_dataset.select(indices=train_index)
train_split.save_to_disk('maxmin-training-spice')

test_split = spice_dataset.select(indices=test_index)
test_split.save_to_disk('maxmin-test-spice')
