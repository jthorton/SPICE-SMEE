# Extract the smiles and energies of the conformers which were kept so we can extract them from the HDF5.
from collections import defaultdict
import pickle

smiles_energies = defaultdict(list)

with open("train_large_neut_no_bad_clean.xyz") as f:
    for line in f.readlines():
        if "Properties" in line:
            ls = line.split()
            if "smiles=" not in line:
                # water config so skip
                continue
            smiles = ls[1].split("smiles=")[1].strip('"')
            energy = float(ls[3].split("energy=")[1])
            config = ls[4].split("config_type=")[1].strip('"')
            print(config)
            if config == "DES370K":
                config += ls[5].strip('"')
            print(config)
            if config in ["PubChem", "Dipeptides", "DES370KMonomers"]:
                print(smiles, energy)
                smiles_energies[smiles].append(energy)
            
    pickle.dump(smiles_energies, open("smiles_and_energies.pickle", "wb"))