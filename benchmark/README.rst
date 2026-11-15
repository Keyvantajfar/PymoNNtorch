==========
Benchmarks
==========

Speed comparison of PymoNNtorch with PymoNNto, Brian2 and NEST on two models:
a LIF network with STDP (``Benchmarks/Simple``) and an Izhikevich network
(``Benchmarks/Izhikevich``), plus raw numpy/torch operation timings
(``Benchmarks/Operations``).

Benchmark code by Ali Rahimi & Marius Vieth.

Running
-------

Each script prints a ``simulation time:`` line. Pass ``no_plot`` to skip plots::

    cd benchmark/Benchmarks/Izhikevich
    python pymonntorch_izh_cpu.py no_plot

``automation.py`` repeats a list of scripts and collects the times in a csv
file (needs ``pandas`` and ``tqdm``)::

    python benchmark/automation.py <repeats> <out.csv | _> <script> [<script> ...]

The Brian2, NEST and PymoNNto scripts need those packages installed; the
``*_cuda.py`` scripts need a CUDA GPU. Previous results are stored in
``Plot/Results``.
