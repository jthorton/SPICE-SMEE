import datetime
import pathlib

import datasets
import tensorboardX
import torch

import descent.optim
import descent.targets.energy
import descent.utils.loss
import descent.utils.reporting
import descent.train
import math
import tqdm
import more_itertools


def write_metrics(
        epoch: int,
        loss: torch.Tensor,
        loss_energy: torch.Tensor,
        loss_forces: torch.Tensor,
        prior_k_torsions: torch.Tensor,
        writer: tensorboardX.SummaryWriter
):
    print(f"epoch={epoch} loss={loss.detach().item():.6f}", flush=True)

    writer.add_scalar("loss", loss.detach().item(), epoch)
    writer.add_scalar("loss_energy", loss_energy.detach().item(), epoch)
    writer.add_scalar("loss_forces", loss_forces.detach().item(), epoch)
    writer.add_scalar("prior_k_torsions", prior_k_torsions.detach().item(), epoch)

    writer.add_scalar("rmse_energy", math.sqrt(loss_energy.detach().item()), epoch)
    writer.add_scalar("rmse_forces", math.sqrt(loss_forces.detach().item()), epoch)
    writer.flush()


def main():
    """Use batching to fit to the SPICE dataset on a single GPU!"""

    n_epochs = 1000
    lr = 0.01
    batch_size = 1000

    dataset_dir = pathlib.Path("datasets/lj-sage-msm-0-torsions/maxmin-training-spice")
    output_dir = pathlib.Path(f"fits")
    fit_dir = output_dir.joinpath("DE-filtered-no-charge-small-mols")
    fit_dir.mkdir(exist_ok=True, parents=True)

    dataset = datasets.Dataset.load_from_disk(dataset_dir.as_posix())
    dataset_size = len(dataset)
    total_configs = sum([len(d["energy"]) for d in dataset])

    force_field, topologies = torch.load("datasets/lj-sage-msm-0-torsions/maxmin-training-spice_ff_top.pt")
    # convert to cuda 
    force_field = force_field.to("cuda")
    topologies = {
        smiles: topology.to("cuda")
        for smiles, topology in topologies.items()
        }

    parameters = {
        "Bonds": descent.train.ParameterConfig(
            cols=["k", "length"],
            scales={"k": 1.0 / 100.0, "length": 1.0},
            limits={"k":[0.0, None], "length": [0.0, None]}
        ),
        "Angles": descent.train.ParameterConfig(
            cols=["k", "angle"],
            scales={"k": 1.0 / 100.0, "angle": 1.0},
            limits={"k": [0.0, None], "angle": [0.0, math.pi]}
        ),
        "ProperTorsions": descent.train.ParameterConfig(
            cols=["k"],
            scales={"k": 1.0},
        ),
        "ImproperTorsions": descent.train.ParameterConfig(
            cols=["k"],
            scales={"k": 1.0},
        )
    }

    trainable = descent.train.Trainable(
        force_field=force_field,
        parameters=parameters,
        attributes={}
    )   

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f"fit-{timestamp}"
    experiment_dir = fit_dir.joinpath(experiment_name)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    x = trainable.to_values().to('cuda')

    with tensorboardX.SummaryWriter(str(experiment_dir)) as writer:
        optimizer = torch.optim.Adam([x], lr=lr, amsgrad=True)

        # write hparams
        for v in tensorboardX.writer.hparams({"optimizer": "Adam", "lr": lr}, {}):
            writer.file_writer.add_summary(v)

        # main fitting loop
        for i in range(n_epochs):

            ff = trainable.to_force_field(x)
            total_loss, energy_loss, force_loss, grad = torch.zeros(size=(1,), device=x.device.type), torch.zeros(size=(1,), device=x.device.type), torch.zeros(size=(1,), device=x.device.type), None
            for batch_ids in tqdm.tqdm(more_itertools.batched([i for i in range(len(dataset))], batch_size),
                                   desc='Calculating energies', ncols=80, total=math.ceil(len(dataset) / batch_size)):
                batch = dataset.select(indices=batch_ids)
                true_batch_size = len(dataset)
                batch_configs = sum([len(d["energy"]) for d in batch])

                e_ref, e_pred, f_ref, f_pred = descent.targets.energy.predict(
                    batch, ff, topologies, "mean"
                )   
                # L2 loss
                batch_loss_energy = ((e_pred - e_ref) ** 2).sum() / true_batch_size
                batch_loss_force = ((f_pred - f_ref) ** 2).sum() / true_batch_size

                # Equal sum of L2 loss on energies and forces
                batch_loss = batch_loss_energy + batch_loss_force

                (batch_grad, ) = torch.autograd.grad(batch_loss, x, create_graph=True)
                batch_grad = batch_grad.detach()
                if grad is None:
                    grad = batch_grad
                else:
                    grad += batch_grad
                
                # keep sum of squares to report MSE at the end
                total_loss += batch_loss.detach()
                energy_loss += batch_loss_energy.detach()
                force_loss += batch_loss_force.detach()
            # print(x.grad) # should be zero?

            # after all batchs add the torsion reg gradient
            k_col_torsion = ff.potentials_by_type['ProperTorsions'].parameter_cols.index("k")
            # average over the workers
            torsion_prior = ff.potentials_by_type['ProperTorsions'].parameters[:, k_col_torsion].square().sum()
            (torsion_grad, ) = torch.autograd.grad(torsion_prior, x, create_graph=False)
            print(grad)
            print(torsion_grad)
            grad += torsion_grad.detach()
            
            # move the grad to the right place
            x.grad = grad
            
            write_metrics(epoch=i, loss=total_loss, loss_energy=energy_loss, loss_forces=force_loss, prior_k_torsions=torsion_prior, writer=writer)

            optimizer.step()
            optimizer.zero_grad()

            if i % 10 == 0:
                torch.save(
                    trainable.to_force_field(x), experiment_dir / f"force-field-epoch-{i}.pt"
                )

    torch.save(trainable.to_force_field(x), experiment_dir / "final-force-field.pt")


if __name__ == "__main__":
    main()



