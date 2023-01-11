# Install AnalogVNN

AnalogVNN is tested and supported on the following 64-bit systems:

- Python 3.7, 3.8, 3.9, 3.10
- Windows 7 and later
- Ubuntu 16.04 and later, including WSL
- Red Hat Enterprise Linux 7 and later
- OpenSUSE 15.2 and later
- macOS 10.12 and later

### Installation

Install [PyTorch](https://pytorch.org/) then:

- Pip:
  ```bash
  # Current stable release for CPU and GPU
  pip install analogvnn
  ```

OR

- AnalogVNN can be downloaded at ([GitHub](https://github.com/Vivswan/AnalogVNN)) or creating a
  fork of it.

<br>

## Dependencies

Install the required dependencies:

- PyTorch
  - Manual installation required: [https://pytorch.org/](https://pytorch.org/)
- dataclasses
- scipy
- numpy
- networkx
- (optional) tensorboard
  - For using tensorboard to visualize the network, with class
    {py:class}`analogvnn.utils.TensorboardModelLog.TensorboardModelLog`
- (optional) torchinfo
  - For adding summary to tensorboard by using
    {py:func}`analogvnn.utils.TensorboardModelLog.TensorboardModelLog.add_summary`
- (optional) graphviz
  - For saving and rendering forward and backward graphs using
    {py:func}`analogvnn.graph.AcyclicDirectedGraph.AcyclicDirectedGraph.render`
- (optional) python-graphviz
  - For saving and rendering forward and backward graphs using
    {py:func}`analogvnn.graph.AcyclicDirectedGraph.AcyclicDirectedGraph.render`

<br>
<br>

That's it, you are all set to simulate analog neural networks.

Head over to the {doc}`tutorial` and look over the {doc}`sample_code`.
