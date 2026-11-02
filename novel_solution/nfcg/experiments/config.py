import itertools

from fcgno.experiments.settings import as_list


# One run per combination of the listed axes of a config
def expand_axes(config, axes):
    return [dict(zip(axes, values)) for values in itertools.product(*[as_list(config[axis]) for axis in axes])]
