import torch

from pymonntorch.NetworkCore.Behavior import Behavior


class Izhikevich_Neuron(Behavior):
    r"""Izhikevich neuron model (Izhikevich, 2003).

    .. math::

        \frac{dv}{dt} &= 0.04 v^2 + 5 v + 140 - u + I \\
        \frac{du}{dt} &= a (b v - u) \\
        &\text{if } v \geq v_{th}: \quad v \leftarrow c, \quad u \leftarrow u + d

    The equations are integrated with the forward Euler method. A spike is
    detected at the beginning of an iteration from the voltage of the previous
    one, after which the reset is applied and the state is advanced by ``dt``.

    The default parameters describe a *regular spiking* cortical neuron. Every
    parameter may be a number or a diversity string (e.g. ``"normal(mean=8, std=1)"``)
    to get a heterogeneous population.

    Note:
        The input current is read from ``neurons.voltage_i``. It is created
        (filled with zeros) by this behavior if no other behavior created it
        before, so synaptic behaviors or :class:`Izhikevich_Neuron_Input` can
        write to it.

    Args:
        a (float): Time scale of the recovery variable ``u``. The default is 0.02.
        b (float): Sensitivity of ``u`` to subthreshold fluctuations of ``v``. The default is 0.2.
        c (float): After-spike reset value of the membrane potential (mV). The default is -65.0.
        d (float): After-spike increment of the recovery variable. The default is 8.0.
        threshold (float): Spike threshold (mV). The default is 30.0.
        dt (float): Euler time step (ms). The default is 0.5.
        init_v (float): Initial membrane potential. The default is ``c``.
        init_u (float): Initial recovery variable. The default is ``b * init_v``.

    Variables set on the neuron group:
        - ``voltage``: membrane potential ``v``.
        - ``u``: recovery variable.
        - ``spike``: boolean tensor, True for neurons that fired in this iteration.
        - ``voltage_i``: input current (if not already present).
    """

    def __init__(
        self,
        *args,
        a=0.02,
        b=0.2,
        c=-65.0,
        d=8.0,
        threshold=30.0,
        dt=0.5,
        init_v=None,
        init_u=None,
        **kwargs,
    ):
        super().__init__(
            *args,
            a=a,
            b=b,
            c=c,
            d=d,
            threshold=threshold,
            dt=dt,
            init_v=init_v,
            init_u=init_u,
            **kwargs,
        )

    def initialize(self, neurons):
        super().initialize(neurons)

        self.a = self.parameter("a", 0.02, neurons)
        self.b = self.parameter("b", 0.2, neurons)
        self.c = self.parameter("c", -65.0, neurons)
        self.d = self.parameter("d", 8.0, neurons)
        self.threshold = self.parameter("threshold", 30.0, neurons)
        self.dt = self.parameter("dt", 0.5, neurons)
        init_v = self.parameter("init_v", None, neurons)
        init_u = self.parameter("init_u", None, neurons)

        neurons.dt = self.dt
        neurons.voltage = neurons.vector() + (self.c if init_v is None else init_v)
        neurons.u = (
            self.b * neurons.voltage
            if init_u is None
            else neurons.vector() + init_u
        )
        neurons.spike = neurons.vector(dtype=torch.bool)

        if not hasattr(neurons, "voltage_i"):
            neurons.voltage_i = neurons.vector()

    def forward(self, neurons):
        v, u = neurons.voltage, neurons.u

        spike = v >= self.threshold
        neurons.spike = spike

        # reset the neurons that fired
        neurons.voltage = v = torch.where(spike, self.c + torch.zeros_like(v), v)
        neurons.u = u = torch.where(spike, u + self.d, u)

        # Euler step; both derivatives use the state *before* the update
        dv = 0.04 * v * v + 5.0 * v + 140.0 - u + neurons.voltage_i
        du = self.a * (self.b * v - u)

        neurons.voltage = v + self.dt * dv
        neurons.u = u + self.dt * du


class Izhikevich_Neuron_Input(Behavior):
    r"""Sets the input current ``neurons.voltage_i`` of Izhikevich neurons.

    The current is ``voltage_i + noise_std * N(0, 1)`` and is redrawn every
    iteration. It *overwrites* ``neurons.voltage_i``, so give it a lower key than
    behaviors that add to the current (e.g. synaptic input).

    Args:
        voltage_i (float): Base (constant) input current. May be a diversity string
            such as ``"normal(mean=10, std=2)"`` for a per-neuron current.
            The default is 0.0.
        scale (float): Multiplier of ``voltage_i``. The default is 1.0.
        noise_std (float): Standard deviation of the Gaussian noise added each
            iteration. The default is 0.0 (no noise).
    """

    def __init__(self, *args, voltage_i=0.0, scale=1.0, noise_std=0.0, **kwargs):
        super().__init__(
            *args, voltage_i=voltage_i, scale=scale, noise_std=noise_std, **kwargs
        )

    def initialize(self, neurons):
        super().initialize(neurons)

        scale = self.parameter("scale", 1.0, neurons)
        self.noise_std = self.parameter("noise_std", 0.0, neurons)
        base = self.parameter("voltage_i", 0.0, neurons)

        neurons.voltage_i = (neurons.vector() + base) * scale
        self.base_current = neurons.voltage_i.clone()

    def forward(self, neurons):
        if isinstance(self.noise_std, torch.Tensor) or self.noise_std != 0.0:
            noise = torch.randn_like(self.base_current) * self.noise_std
            neurons.voltage_i = self.base_current + noise
        else:
            neurons.voltage_i = self.base_current
