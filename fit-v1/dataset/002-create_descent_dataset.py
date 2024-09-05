import h5py
from openff.units import unit
import pickle
from qcengine.units import ureg
import numpy
import tqdm
import descent.targets.energy

smiles_energies = pickle.load(open("smiles_and_energies.pickle", "rb"))

all_data = []
hartree_to_kcal = ureg.conversion_factor("hartree", "kcal/mol")

with h5py.File("SPICE-1.1.4.hdf5") as spice:
    for entry, record in tqdm.tqdm(spice.items(), desc="Extracting dataset", ncols=80):
        smiles = record["smiles"].asstr()[0]
        subset = record["subset"].asstr()[0]
        # only extract the data if it was in the filtered MACE list
        if smiles in smiles_energies:
            # awkward number conversions mean we need to round a little
            spice_energies_round = numpy.round(record["dft_total_energy"][:], decimals=5)
            ref_energies = numpy.round(numpy.array(smiles_energies[smiles]) * ureg.conversion_factor("eV", "hartree"), decimals=5)
            samples = [i for i in range(len(spice_energies_round)) if spice_energies_round[i] in ref_energies]
            # extract the data 
            energies = [record["dft_total_energy"][i] * hartree_to_kcal for i in samples]
            coords = [record["conformations"][i] * ureg.conversion_factor("bohr", "angstrom") for i in samples]
            forces = [record["dft_total_gradient"][i] * -1 * (hartree_to_kcal / ureg.conversion_factor("bohr", "angstrom")) for i in samples]
            all_data.append(
                {
                    "smiles": smiles,
                    "coords": coords,
                    "energy": energies,
                    "forces": forces
                }
            )
    print("creating descent dataset")
    dataset = descent.targets.energy.create_dataset(entries=all_data)
    dataset.save_to_disk("mace_filtered_spice_114")




