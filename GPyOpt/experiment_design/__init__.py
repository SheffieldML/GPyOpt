from .base import ExperimentDesign
from .grid_design import GridDesign
from .latin_design import LatinDesign
from .random_design import RandomDesign
from .sobol_design import SobolDesign

def initial_design(design_name, space, init_points_count):
    design = None
    if design_name == 'random':
        design = RandomDesign(space)
    elif design_name == 'sobol':
        design = SobolDesign(space)
    elif design_name == 'grid':
        design = GridDesign(space)
    elif design_name == 'latin':
        design = LatinDesign(space)
    else:
        raise ValueError('Unknown design type: ' + design_name)

    return design.get_samples(init_points_count)