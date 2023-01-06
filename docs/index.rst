Welcome to AnalogVNN's documentation!
=====================================

**AnalogVNN** is a simulation framework built on PyTorch which can simulate the effects of
analog components like optoelectronic noise, limited precision, and signal normalization
present in photonics neural network accelerators. By following the same layer structure
design present in PyTorch, the AnalogVNN framework allows users to convert most
digital neural network models to their analog counterparts with just a few lines of
code, taking full advantage of the open-source optimization, deep learning, and GPU
acceleration libraries available through PyTorch.

Citing AnalogVNN
_________________
We would appreciate if you cite the following paper in your publications for which you used AnalogVNN:

Vivswan Shah, and Nathan Youngblood. "AnalogVNN: A fully modular framework for modeling and optimizing photonic neural networks." *arXiv preprint arXiv:2210.10048 (2022)*.

Table of contents
_________________

.. toctree::
   :maxdepth: 2

   install
   sample_code
   tutorial
   extra_classes
   inner_workings

.. only:: html

   Indices and tables
   ==================

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
