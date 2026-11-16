"""Real-time visualization of a small feed-forward network.

Needs the visualizer extras:  pip install "pymonntorch[viz]"
(and "pymonntorch[viz-cuda]" together with ``--device cuda``).

    python examples/visualizer_gui.py            # CPU
    python examples/visualizer_gui.py --device cuda
"""

import argparse

import torch

from pymonntorch import (
    Behavior,
    Network,
    NeuronDimension,
    NeuronGroup,
    SynapseGroup,
)


class RandomSpikes(Behavior):
    def initialize(self, ng):
        super().initialize(ng)
        ng.spikes = ng.vector("rand") > 0.5

    def forward(self, ng):
        ng.spikes = ng.vector("rand") > 0.5


class Trace(Behavior):
    def initialize(self, ng):
        super().initialize(ng)
        self.tau = self.parameter("tau", 10.0, ng)
        ng.trace = ng.vector()

    def forward(self, ng):
        # in-place update: the visualizer keeps a view of this tensor
        ng.trace += ng.spikes * 0.25 - ng.trace / self.tau


class SpikeOnTrace(Behavior):
    def initialize(self, sg):
        super().initialize(sg)
        sg.weights = sg.matrix("uniform") / sg.dst.size / 5

    def forward(self, sg):
        sg.dst.trace += torch.sum(sg.weights[sg.src.spikes], axis=0)


def build_network(device="cpu"):
    net = Network(dtype=torch.float32, device=device)

    behavior = {1: RandomSpikes(), 3: Trace(tau=10)}
    groups = [
        NeuronGroup(
            net=net,
            size=NeuronDimension(depth=depth, height=height, width=width),
            behavior=dict(behavior),
        )
        for depth, height, width in [(1, 100, 100), (4, 20, 20), (1, 10, 10), (15, 2, 2)]
    ]
    for src, dst in zip(groups[:-1], groups[1:]):
        SynapseGroup(net=net, src=src, dst=dst, behavior={4: SpikeOnTrace()})

    net.initialize()
    return net


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    from pymonntorch.visualizer import GUI

    GUI(build_network(args.device), variable="trace").initializeOpenGL()


if __name__ == "__main__":
    main()
