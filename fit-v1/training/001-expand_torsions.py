import click
from openff.toolkit import ForceField
from openff.units import unit


@click.command()
@click.option('-ff', '--forcefield')
@click.option('-o', '--output')
def main(forcefield: str, output: str):
    """
    Expand the torsions in the base force field for a valence fit and set all k terms to 0.
    """
    ff = ForceField(forcefield, load_plugins=True)
    torsion_handler = ff.get_parameter_handler('ProperTorsions')
    for parameter in torsion_handler.parameters:
        parameter.idivf = [1.0] * 4
        parameter.k = [0 * unit.kilocalories_per_mole] * 4
        parameter.periodicity = [1, 2, 3, 4]
        parameter.phase = [0 * unit.degree, 180 * unit.degree, 0 * unit.degree, 180 * unit.degree]

    improper_handler = ff.get_parameter_handler('ImproperTorsions')
    # reset all improper to 0 as well 
    for parameter in improper_handler.parameters:
        parameter.k = [0 * unit.kilocalories_per_mole]

    ff.to_file(output)


if __name__ == '__main__':
    main()