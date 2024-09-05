import functools
import multiprocessing
import pathlib
import openff.interchange
import openff.toolkit
import smee
import smee.converters
import torch
import tqdm
import datasets


def build_interchange(
        smiles: str, force_field_path: str
) -> openff.interchange.Interchange | None:
    try:
        return openff.interchange.Interchange.from_smirnoff(
            force_field=openff.toolkit.ForceField(force_field_path),
            topology=openff.toolkit.Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True).to_topology()
            )
    except BaseException as e:
        print(f"Failed to parameterize {smiles}: {e}")
        return None


def apply_parameters(unique_smiles: list[str], force_field_path: str) -> tuple[smee.TensorForceField, dict[str, smee.TensorTopology]]:
    build_fn = functools.partial(
        build_interchange, force_field_path=force_field_path
    )

    with multiprocessing.get_context("spawn").Pool() as pool:
        interchanges = list(
            tqdm.tqdm(
                pool.imap(build_fn, unique_smiles),
                total=len(unique_smiles),
                desc="Building interchanges",
                ncols=80
            )
        )
    unique_smiles, interchanges = zip(*[(s, i) for s, i in zip(unique_smiles, interchanges) if i is not None])

    force_field, topologies = smee.converters.convert_interchange(interchanges)

    return force_field, {smiles: topology for smiles, topology in zip(unique_smiles, topologies)}


def main():

    force_field_name = "../input_ff/lj-sage-msm-0-torsions.offxml"
    output_dir = pathlib.Path("lj-sage-msm-0-torsions")
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in ["maxmin-training-spice", "maxmin-test-spice"]:
        dataset = datasets.load_from_disk(dataset_name)

        unique_smiles = set()

        for entry in dataset:
            unique_smiles.add(entry["smiles"])

        print(f"N smiles={len(unique_smiles)} in {dataset_name}", flush=True)

        force_field, topologies = apply_parameters(unique_smiles, force_field_name)

        torch.save((force_field, topologies), output_dir.joinpath(f"{dataset_name}_ff_top.pt"))


if __name__ == "__main__":
    main()
