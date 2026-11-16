"""Optional real-time visualizer for PymoNNtorch networks.

Needs the extra dependencies: ``pip install "pymonntorch[viz]"``.

The heavy imports (OpenGL, GLFW, ImGui) happen lazily, so importing
``pymonntorch`` itself never requires them.
"""

__all__ = ["GUI"]


def __getattr__(name):
    if name == "GUI":
        try:
            from pymonntorch.visualizer.Visualize_OpenGL_IMGUI import GUI
        except ImportError as error:
            raise ImportError(
                "The visualizer needs extra dependencies. "
                'Install them with: pip install "pymonntorch[viz]"'
            ) from error
        return GUI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
