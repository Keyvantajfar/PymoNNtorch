"""Smoke tests: the example scripts run and produce spikes."""

import importlib.util
import pathlib

import matplotlib
matplotlib.use("Agg")

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"


def load_example(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_population_example():
    net = load_example("izhikevich_single_population").main(
        iterations=300, size=20, show=False
    )
    assert len(net["spike.t", 0]) > 0


def test_two_populations_example_synapse_drives_destination():
    module = load_example("izhikevich_two_populations")
    uncoupled = module.main(iterations=300, size=8, weight=0.0, show=False)
    coupled = module.main(iterations=300, size=8, weight=20.0, show=False)

    assert len(coupled["spike.t", 0]) > 0
    # the source population is unaffected, the destination fires more when coupled
    assert len(coupled["spike.t", 0]) == len(uncoupled["spike.t", 0])
    assert len(coupled["spike.t", 1]) > len(uncoupled["spike.t", 1])
