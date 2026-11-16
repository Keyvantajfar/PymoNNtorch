=======
History
=======

Unreleased
----------

* Izhikevich neuron model (`Izhikevich_Neuron`, `Izhikevich_Neuron_Input`).
* Examples directory.
* Optional real-time OpenGL visualizer (`pymonntorch.visualizer`, install with `pip install "pymonntorch[viz]"`).
* Benchmarks restored in `benchmark/`.
* Fixes: `all_sub_structure` now includes direct children, missing `sympy` dependency, test suite updated to current signatures.
* GitHub Actions test workflow replaces Travis CI.

0.1.4 (2024-03-16)
------------------

* Bug fixes.
* Src and dst can be `None` for syanpses.
* Hierarchical structure.

0.1.3 (2023-08-16)
------------------

* BREAKING CHANGE: `Network` no longer accept settings. Individual setting are now argument for Network.
* Bug fixes.


0.1.2 (2023-06-14)
------------------

* `tensor` method for NetworkObject


0.1.1 (2023-05-26)
------------------

* Every NetworkObject can have a recorder behavior.
* Network settings accept "index" entry.
* Bug fixes and general improvement.


0.1.0 (2023-03-17)
------------------

* Repository made public.
