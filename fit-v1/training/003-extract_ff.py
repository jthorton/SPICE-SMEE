import click
from openff.toolkit import ForceField
import torch


@click.command()
@click.option('-ff', '--force-field')
@click.option('-tf', '--tensor-field')
@click.option('-o', '--output')
def main(force_field: str, output: str, tensor_field: str):
    """Try and convert the valence fit force field parameters onto the base ff."""

    tensor_ff = torch.load(tensor_field)

    base_ff = ForceField(force_field, load_plugins=True, allow_cosmetic_attributes=True)

    for potential in tensor_ff.potentials:
        ff_name = potential.parameter_keys[0].associated_handler

        parameter_names = potential.parameter_cols
        parameter_units = potential.parameter_units

        if ff_name in ["Bonds", "Angles"]:
            handler = base_ff.get_parameter_handler(ff_name)
            for i in range(len(potential.parameters)):
                smirks = potential.parameter_keys[i].id
                ff_parameter = handler[smirks]
                opt_parameters = potential.parameters[i].detach().cpu().numpy()
                for j, (p, unit) in enumerate(zip(parameter_names, parameter_units)):
                    setattr(ff_parameter, p, opt_parameters[j] * unit)

        elif ff_name in ["ProperTorsions"]:
            handler = base_ff.get_parameter_handler(ff_name)
            # we need to collect the k values into a list across the entries
            collection_data = {}
            for i in range(len(potential.parameters)):
                smirks = potential.parameter_keys[i].id
                if smirks not in collection_data:
                    collection_data[smirks] = [0, 0, 0, 0]
                opt_parameters = potential.parameters[i].detach().cpu().numpy()
                # find k and the periodicity
                k_index = parameter_names.index('k')
                k = opt_parameters[k_index] * parameter_units[k_index]
                p = int(opt_parameters[parameter_names.index('periodicity')]) - 1
                collection_data[smirks][p] = k
            # now update the force field
            for smirks, k_s in collection_data.items():
                ff_parameter = handler[smirks]
                ff_parameter.k = k_s

        elif ff_name in ["ImproperTorsions"]:
            handler = base_ff.get_parameter_handler(ff_name)
            # we only fit the v2 terms for improper torsions so convert to list and set
            for i in range(len(potential.parameters)):
                smirks = potential.parameter_keys[i].id
                opt_parameters = potential.parameters[i].detach().cpu().numpy()
                k_index = parameter_names.index('k')
                ff_parameter = handler[smirks]
                ff_parameter.k = [opt_parameters[k_index] * parameter_units[k_index]]

    base_ff.to_file(output)


if __name__ == '__main__':
    main()
