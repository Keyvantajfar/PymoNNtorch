"""Two Izhikevich populations connected by a synapse group.

The synapse behavior below adds the weighted spikes of the source population
to the input current of the destination population.
"""

import matplotlib.pyplot as plt
import torch

from pymonntorch import (
    Behavior,
    EventRecorder,
    Izhikevich_Neuron,
    Izhikevich_Neuron_Input,
    Network,
    NeuronGroup,
    Recorder,
    SynapseGroup,
)


class SpikeCurrent(Behavior):
    """Adds `weight * (number of presynaptic spikes)` to the destination current."""

    def initialize(self, synapse):
        super().initialize(synapse)
        self.weight = self.parameter("weight", 5.0, synapse)
        synapse.W = synapse.matrix("uniform") * self.weight

    def forward(self, synapse):
        # W has shape (src, dst) when the network uses the SxD matrix mode (default)
        current = synapse.src.spike.to(synapse.W.dtype) @ synapse.W
        synapse.dst.voltage_i = synapse.dst.voltage_i + current


def main(iterations=1000, size=10, weight=20.0, show=True):
    torch.manual_seed(73)
    net = Network(dtype=torch.float32, device="cpu")

    NeuronGroup(
        net=net,
        tag="pop0",
        size=size,
        behavior={
            1: Izhikevich_Neuron(),
            2: Izhikevich_Neuron_Input(voltage_i="normal(mean=10, std=3)"),
            9: Recorder(["voltage"]),
            10: EventRecorder(["spike"]),
        },
    )
    NeuronGroup(
        net=net,
        tag="pop1",
        size=size,
        behavior={
            2: Izhikevich_Neuron_Input(voltage_i=3.0, noise_std=1.0),
            # synaptic input is added after the input behavior set the base current
            3: Izhikevich_Neuron(),
            9: Recorder(["voltage"]),
            10: EventRecorder(["spike"]),
        },
    )
    SynapseGroup(net=net, src="pop0", dst="pop1", tag="GLUTAMATE", behavior={
        2: SpikeCurrent(weight=weight),
    })

    net.initialize()
    net.simulate_iterations(iterations)

    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(10, 6))
    for col, pop in enumerate(("pop0", "pop1")):
        axes[0, col].plot(net["voltage", col][:, :5].cpu())
        axes[0, col].set_title(pop)
        axes[1, col].plot(
            net["spike.t", col].cpu(), net["spike.i", col].cpu(), ".k", markersize=3
        )
    axes[0, 0].set_ylabel("voltage (mV)")
    axes[1, 0].set_ylabel("neuron")
    fig.tight_layout()
    if show:
        plt.show()
    return net


if __name__ == "__main__":
    main()
