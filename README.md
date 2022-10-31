# AnalogVNN

AnalogVNN Paper: [https://arxiv.org/abs/2210.10048](https://arxiv.org/abs/2210.10048)

Documentation: [https://analogvnn.readthedocs.io/](https://analogvnn.readthedocs.io/)

AnalogVNN is a simulation framework built on PyTorch which can simulate the effects of
optoelectronic noise, limited precision, and signal normalization present in photonic
neural network accelerators. We use this framework to train and optimize linear and
convolutional neural networks with up to 9 layers and ~1.7 million parameters, while
gaining insights into how normalization, activation function, reduced precision, and
noise influence accuracy in analog photonic neural networks. By following the same layer
structure design present in PyTorch, the AnalogVNN framework allows users to convert most
digital neural network models to their analog counterparts with just a few lines of code,
taking full advantage of the open-source optimization, deep learning, and GPU acceleration
libraries available through PyTorch.
