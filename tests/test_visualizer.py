"""The visualizer is optional: importing pymonntorch must not need OpenGL."""

import importlib.util
import pathlib
import sys

import pytest


def test_core_import_does_not_load_visualizer_dependencies():
    import pymonntorch  # noqa: F401

    for heavy in ("OpenGL", "glfw", "imgui_bundle"):
        assert heavy not in sys.modules


def test_gui_import_gives_helpful_error_or_works():
    import pymonntorch.visualizer as viz

    try:
        gui = viz.GUI
    except ImportError as error:
        assert "pymonntorch[viz]" in str(error)
    else:
        assert gui.__name__ == "GUI"


def test_unknown_attribute():
    import pymonntorch.visualizer as viz

    with pytest.raises(AttributeError):
        viz.DoesNotExist


def test_example_network_builds_without_gui():
    path = pathlib.Path(__file__).resolve().parent.parent / "examples" / "visualizer_gui.py"
    spec = importlib.util.spec_from_file_location("visualizer_gui", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    net = module.build_network("cpu")
    net.simulate_iterations(5)
    assert len(net.NeuronGroups) == 4
    assert net.NeuronGroups[0].trace.shape == (10000,)
