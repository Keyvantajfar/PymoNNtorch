"""Tests for the Izhikevich neuron behaviors."""

import pytest
import torch

from pymonntorch import (
    Network,
    NeuronGroup,
    EventRecorder,
    Recorder,
    Izhikevich_Neuron,
    Izhikevich_Neuron_Input,
)


def make_net(size=5, neuron_kwargs=None, input_kwargs=None, recorders=True):
    net = Network(dtype=torch.float64)
    behavior = {
        1: Izhikevich_Neuron(**(neuron_kwargs or {})),
        2: Izhikevich_Neuron_Input(**(input_kwargs or {})),
    }
    if recorders:
        behavior[9] = Recorder(["voltage", "u"])
        behavior[10] = EventRecorder(["spike"])
    NeuronGroup(net=net, size=size, tag="ng", behavior=behavior)
    net.initialize(info=False)
    return net


def reference(iterations, a, b, c, d, threshold, dt, current):
    """Plain-python Euler reference for a single neuron."""
    v, u = c, b * c
    vs, spikes = [], []
    for t in range(iterations):
        vs.append(v)
        if v >= threshold:
            spikes.append(t)
            v, u = c, u + d
        dv = 0.04 * v * v + 5 * v + 140 - u + current
        du = a * (b * v - u)
        v, u = v + dt * dv, u + dt * du
    return vs, spikes


def test_single_euler_step_by_hand():
    # v=-65, u=b*v=-13, I=10: dv = 169 - 325 + 140 + 13 + 10 = 7, du = 0
    net = make_net(size=1, input_kwargs={"voltage_i": 10.0}, recorders=False)
    net.simulate_iteration()
    assert net.ng.voltage.item() == pytest.approx(-65.0 + 0.5 * 7.0)
    assert net.ng.u.item() == pytest.approx(-13.0)


def test_matches_reference_implementation():
    n_iter = 400
    net = make_net(size=1, input_kwargs={"voltage_i": 10.0})
    net.simulate_iterations(n_iter)
    ref_v, ref_spikes = reference(n_iter, 0.02, 0.2, -65.0, 8.0, 30.0, 0.5, 10.0)

    # the recorder stores the voltage *after* each iteration, i.e. ref_v[1:] + final
    rec_v = net["voltage", 0][:, 0].tolist()
    assert rec_v[: n_iter - 1] == pytest.approx(ref_v[1:], rel=1e-9)

    spike_t = net["spike.t", 0].tolist()
    assert len(spike_t) == len(ref_spikes) > 0


def test_parameters_are_used():
    slow = make_net(size=1, neuron_kwargs={"d": 2.0}, input_kwargs={"voltage_i": 10.0})
    fast = make_net(size=1, neuron_kwargs={"d": 8.0}, input_kwargs={"voltage_i": 10.0})
    slow.simulate_iterations(600)
    fast.simulate_iterations(600)
    # a smaller after-spike increment of u lets the neuron fire more often
    assert len(slow["spike.t", 0]) > len(fast["spike.t", 0])


def test_no_input_stays_at_rest():
    net = make_net(size=3, input_kwargs={"voltage_i": 0.0})
    net.simulate_iterations(200)
    assert len(net["spike.t", 0]) == 0
    # without input the neuron relaxes to its stable fixed point (-70 mV for the defaults)
    assert torch.allclose(
        net.ng.voltage, torch.full((3,), -70.0, dtype=torch.float64), atol=0.5
    )


def test_voltage_i_created_without_input_behavior():
    net = Network(dtype=torch.float64)
    NeuronGroup(net=net, size=2, tag="ng", behavior={1: Izhikevich_Neuron()})
    net.initialize(info=False)
    net.simulate_iterations(5)
    assert net.ng.voltage_i.shape == (2,)


def test_noise_makes_current_vary():
    net = make_net(size=4, input_kwargs={"voltage_i": 5.0, "noise_std": 1.0})
    net.simulate_iteration()
    first = net.ng.voltage_i.clone()
    net.simulate_iteration()
    assert not torch.equal(first, net.ng.voltage_i)


def test_heterogeneous_parameter_via_diversity_string():
    net = make_net(
        size=6,
        neuron_kwargs={"d": "normal(mean=8, std=1)"},
        input_kwargs={"voltage_i": 10.0},
        recorders=False,
    )
    net.simulate_iterations(10)
    assert net.ng.voltage.shape == (6,)
    assert net.ng.u.unique().numel() > 1  # neurons really differ
