# SPICE-SMEE-v1

This directory contains the scripts needed to prepare the subset of SPICE-1.4 and train a SMIRNOFF style force field to
the energies and forces using SMEE and DESCENT.

## Plan 

Fit to a neutral filtered subset of SPICE-1.1.4 drawing from the DESMonomers, Dipeptides and PubChem sets which have forces
that agree with the MACE-OFF-Large model. As the MACE models are able to recover accurate torsions from the SPICE dataset
this may mean we can use this type of data to train a MM force field without expensive torsion drives.

# Dataset Generation

These scripts detail how to generate the neutral subset of SPICE used in the fit.
First download the SPICE-1.1.4 HDF5 file from [Zenodo](https://zenodo.org/records/8222043), you will also need the 
filtered conformers (`train_large_neut_no_bad_clean.tar.gz`) used to fit the large MACE-OFF model which can be found 
[here](https://www.repository.cam.ac.uk/items/d50227cd-194f-4ba4-aeb7-2643a69f025f).

You should then use each of the scripts in the dataset folder to generate the final dataset used for training.

## Dataset Scripts:
- `001-extract_smiles_energies.py`: Extract the smiles and energies of the conformers which should be saved from the SPICE dataset.
- `002-create_descent_dataset.py`: Using the extracted smiles and energies build a descent energy and force dataset which can be used for fitting.
- `003-split_dataset.py`: Use the deepchem maxmin diversity spliter to split the filtered SPICE dataset, note all molecules are mixed together rather than splitting by subset.
- `004-make_datasets.py`: Based on the maxmin splits build the training and testing descent dataset.
- `005-parameterize.py`: Try and parameterise the training and testing descent datasets and save the topologies and force field for fitting.
- `006-filter_dataset_issues.py`: Remove molecules from the training and testing datasets which could not be parameterised (issues include missing bccs, conformer generation failures for charges).


# Training

These scripts detail how the training was conducted including any pre-processing which was done to the force field. Note we start from the 
Sage-2.2.0 MSM starting point to avoid any bias from previous fits to ForceBalance style targets. 

We have also included the initial force field: `lj-sage-msm-0-torsions.offxml` and the final force field after 740 epochs
of training `final_ff.offxml`.


## Training Scripts

- `001-expand_torsions.py`: Used to expand the proper torsion terms to include all periodicities 1-4 and set them to 0. Improper torsions are kept at k2 but set to 0.
- `002-train-valence-adam.py`: Used to train the force field using the GPU and batching.
- `003-extract_ff.py`: Used to extract the final valence parameters from the optimised tensor force field and create an updated base offxml with the values. 