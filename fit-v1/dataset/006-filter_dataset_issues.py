# filter the datasets to remove molecules that could not be parameterized
import datasets
import torch
import pathlib


def main():
    top_dir = pathlib.Path("lj-sage-msm-0-torsions")

    for dataset_name, top_name in [("maxmin-training-spice", "maxmin-training-spice_ff_top.pt"), ("maxmin-test-spice", "maxmin-test-spice_ff_top.pt")]:
        dataset = datasets.load_from_disk(dataset_name)        
        _, topologies = torch.load(top_dir.joinpath(top_name))

        dataset_size = len(dataset)
        dataset = dataset.filter(lambda x: x["smiles"] in topologies)

        print(f"Removed issues : {dataset_size} -> {len(dataset)}")

        # write the dataset to the new folder
        dataset.save_to_disk(top_dir.joinpath(dataset_name).as_posix())


if __name__ == "__main__":
    main()

