"""Main module."""
from pymonntorch import *

torch.manual_seed(73)
settings = {"def_type": torch.float32, "device": 'cpu'} #, "synapse_mode": (SxD)}

net = Network(settings=settings)

pop0 = NeuronGroup(net = net, tag = "pop0", size =10 ,
                   behavior ={1: Izhikevich_Neuron(), 2: Izhikevich_Neuron_Input(voltage_i=10.0), 
                              9: Recorder (['voltage']), 10: EventRecorder (['spike'])})
pop1 = NeuronGroup(net = net, tag = "pop1", size =10 ,
                   behavior ={1: Izhikevich_Neuron(), 2: Izhikevich_Neuron_Input(voltage_i=10.0), 
                              9: Recorder (['voltage']), 10: EventRecorder (['spike'])})

SynapseGroup(pop0, pop1, net)

net.initialize()
net.simulate_iterations(1000)
import matplotlib . pyplot as plt

plt.plot(net["voltage", 0][:, :10]), plt.show()
try:
    plt.plot(net["u", 0][:, :10]), plt.show()
except Exception as e:
    print(f"Error >> {e}  << happened")
finally:
    plt.plot(net["spike.t", 0],net["spike.i", 0],'.k')
    plt.show()