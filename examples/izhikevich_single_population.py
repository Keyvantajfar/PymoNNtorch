"""A single population of Izhikevich neurons driven by a random constant current."""

import matplotlib.pyplot as plt
import torch

from pymonntorch import (
    EventRecorder,
    Izhikevich_Neuron,
    Izhikevich_Neuron_Input,
    Network,
    NeuronGroup,
    Recorder,
)


def main(iterations=1000, size=100, show=True):
    torch.manual_seed(73)
    net = Network(dtype=torch.float32, device="cpu")
    NeuronGroup(
        net=net,
        tag="pop",
        size=size,
        behavior={
            1: Izhikevich_Neuron(),
            # a different constant current for every neuron
            2: Izhikevich_Neuron_Input(voltage_i="normal(mean=10, std=3)"),
            9: Recorder(["voltage", "u"]),
            10: EventRecorder(["spike"]),
        },
    )
    net.initialize()
    net.simulate_iterations(iterations)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    axes[0].plot(net["voltage", 0][:, :5].cpu())
    axes[0].set_ylabel("voltage (mV)")
    axes[1].plot(net["u", 0][:, :5].cpu())
    axes[1].set_ylabel("recovery u")
    axes[2].plot(net["spike.t", 0].cpu(), net["spike.i", 0].cpu(), ".k", markersize=2)
    axes[2].set_ylabel("neuron")
    axes[2].set_xlabel("iteration")
    fig.tight_layout()
    if show:
        plt.show()
    return net


if __name__ == "__main__":
    main()
