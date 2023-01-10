from __future__ import annotations

import dataclasses
import typing
import warnings
from collections import namedtuple
from dataclasses import dataclass
from distutils.version import LooseVersion
from typing import Optional, Sequence, List, Dict, Union, Any, Callable

import torch
from torch import Tensor, nn

from analogvnn.nn.module.Layer import Layer
from analogvnn.utils.is_cpu_cuda import is_cpu_cuda

if typing.TYPE_CHECKING:
    from graphviz import Digraph

__all__ = [
    'save_autograd_graph',
    'save_autograd_graph_from_trace',
    'get_autograd_dot',
    'get_autograd_dot_from_trace',
    'get_autograd_obj',
    'make_autograd_obj',
    'size_to_str',
    'AutoGradDot',
]

Node = namedtuple('Node', ('name', 'inputs', 'attr', 'op'))

# Saved attrs for grad_fn (incl. saved variables) begin with `._saved_*`
SAVED_PREFIX = "_saved_"


def size_to_str(size):
    """Convert a tensor size to a string.

    Args:
        size (torch.Size): the size to convert.

    Returns:
        str: the string representation of the size.
    """
    return '(' + ', '.join(['%d' % s for s in size]) + ')'


def resize_graph(dot: Digraph, size_per_element: float = 0.15, min_size: float = 12):
    """Resize the graph according to how much content it contains.

    Modify the graph in place.

    Args:
        dot (graphviz.Digraph): graph to be resized
        size_per_element (float): A "rank" in graphviz contains roughly
            size_per_element**2 pixels.
        min_size (float): Minimum size of graph.
    """
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)


def get_fn_name(fn: Callable, show_attrs: bool, max_attr_chars: int) -> str:
    """Get the name of a function.

    Args:
        fn (Callable): the function.
        show_attrs (bool): whether to show the attributes.
        max_attr_chars (int): the maximum number of characters to show for the attributes.

    Returns:
        str: the name of the function.
    """

    name = str(type(fn).__name__)
    if not show_attrs:
        return name
    attrs = dict()
    for attr in dir(fn):
        if not attr.startswith(SAVED_PREFIX):
            continue
        val = getattr(fn, attr)
        attr = attr[len(SAVED_PREFIX):]
        if torch.is_tensor(val):
            attrs[attr] = "[saved tensor]"
        elif isinstance(val, tuple) and any(torch.is_tensor(t) for t in val):
            attrs[attr] = "[saved tensors]"
        else:
            attrs[attr] = str(val)
    if not attrs:
        return name
    max_attr_chars = max(max_attr_chars, 3)
    col1width = max(len(k) for k in attrs.keys())
    col2width = min(max(len(str(v)) for v in attrs.values()), max_attr_chars)
    sep = "-" * max(col1width + col2width + 2, len(name))
    attrstr = '%-' + str(col1width) + 's: %' + str(col2width) + 's'
    truncate = lambda s: s[:col2width - 3] + "..." if len(s) > col2width else s
    params = '\n'.join(attrstr % (k, truncate(str(v))) for (k, v) in attrs.items())
    return name + '\n' + sep + '\n' + params


@dataclass
class AutoGradDot:
    """Stores and manages Graphviz representation of PyTorch autograd graph.

    Attributes:
        dot (graphviz.Digraph): Graphviz representation of the autograd graph.
        _module (nn.Module): The module to be traced.
        _inputs (List[Tensor]): The inputs to the module.
        _inputs_kwargs (Dict[str, Tensor]): The keyword arguments to the module.
        _outputs (Sequence[Tensor]): The outputs of the module.
        param_map (Dict[int, str]): A map from parameter values to their names.
        _seen (set): A set of nodes that have already been added to the graph.
        show_attrs (bool): whether to display non-tensor attributes of backward nodes
            (Requires PyTorch version >= 1.9)
        show_saved (bool): whether to display saved tensor nodes that are not by custom
            autograd functions. Saved tensor nodes for custom functions, if
            present, are always displayed. (Requires PyTorch version >= 1.9)
        max_attr_chars (int): if show_attrs is `True`, sets max number of characters
            to display for any given attribute.
        _called (bool): the module has been called.
    """

    dot: Digraph = None
    _module: nn.Module = None

    _inputs: Sequence[Tensor] = dataclasses.field(default=None, repr=False, hash=False)
    _inputs_kwargs: Dict[str, Tensor] = dataclasses.field(default_factory=dict, repr=False, hash=False)
    _outputs: Sequence[Tensor] = dataclasses.field(default=None, repr=False, hash=False)

    param_map: dict = dataclasses.field(default_factory=dict, repr=False, hash=False)
    _seen: set = dataclasses.field(default_factory=set, repr=False, hash=False)

    show_attrs: bool = dataclasses.field(default=False, repr=False, hash=False)
    show_saved: bool = dataclasses.field(default=False, repr=False, hash=False)
    max_attr_chars: int = dataclasses.field(default=50, repr=False, hash=False)
    _called: bool = False

    def __post_init__(self):
        """Create the graphviz graph.

        Raises:
            ImportError: if graphviz (https://pygraphviz.github.io/) is not available.
        """

        try:
            from graphviz import Digraph
        except ImportError as e:
            raise ImportError("requires graphviz: https://pygraphviz.github.io/") from e

        node_attr = dict(
            style='filled',
            shape='box',
            align='left',
            fontsize='12',
            ranksep='0.1',
            height='0.2',
            fontname='monospace'
        )
        self.dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"), format="svg")

    def __call__(self) -> Optional[Sequence[Tensor]]:
        """Call the module with inputs and returns the outputs.

        Returns:
            Optional[Sequence[Tensor]]: the outputs of the module.
        """

        assert self._module is not None
        outputs = self._module(*self.inputs, **self.inputs_kwargs)

        if outputs is None:
            return outputs

        if isinstance(outputs, tuple):
            return outputs

        return (outputs,)

    @property
    def inputs(self) -> Sequence[Tensor]:
        """the arg inputs to the module

        Returns:
            Sequence[Tensor]: the arg inputs to the module.
        """
        return self._inputs

    @inputs.setter
    def inputs(self, inputs: Union[Tensor, Sequence[Tensor]]):
        """Set the inputs to the module.

        Args:
            inputs (Union[Tensor, Sequence[Tensor]]): the inputs to the module.
        """
        self._inputs = inputs
        self._called = False

        if not inputs:
            return

        if not isinstance(inputs, tuple):
            inputs = (inputs,)

        for i, v in enumerate(inputs):
            self.param_map[id(v)] = f"INPUT_{i}"
            self.param_map[id(v.data)] = f"INPUT_{i}"

    @property
    def inputs_kwargs(self) -> Dict[str, Tensor]:
        """the keyword inputs to the module

        Args:
            Dict[str, Tensor]: the keyword inputs to the module.
        """
        return self._inputs_kwargs

    @inputs_kwargs.setter
    def inputs_kwargs(self, inputs_kwargs: Dict[str, Tensor]):
        """Set the keyword inputs to the module.

        Args:
            inputs_kwargs (Dict[str, Tensor]): the keyword inputs to the module.
        """
        self._inputs_kwargs = inputs_kwargs
        self._called = False

        if not inputs_kwargs:
            return

        for k, v in inputs_kwargs.items():
            self.param_map[id(v)] = f"INPUT_{k}"
            self.param_map[id(v.data)] = f"INPUT_{k}"

    @property
    def outputs(self) -> Optional[Sequence[Tensor]]:
        """the outputs of the module

        Returns:
            Optional[Sequence[Tensor]]: the outputs of the module.
        """
        if self._called:
            return self._outputs

        outputs = self.__call__()
        self._called = True

        if not outputs:
            return

        for i, v in enumerate(outputs):
            self.param_map[id(v)] = f"OUTPUT_{i}"
            self.param_map[id(v.data)] = f"OUTPUT_{i}"

        self._outputs = outputs
        return self._outputs

    @property
    def module(self):
        """the module.

        Returns:
            nn.Module: the module to be traced.
        """
        return self._module

    @module.setter
    def module(self, module: nn.Module):
        """Set the module.

        Args:
            module (nn.Module): the module.
        """
        self._module = module
        self._called = False

        if not module:
            return

        assert isinstance(module, nn.Module)
        for k, v in dict(module.named_parameters()).items():
            self.param_map[id(v)] = k
            self.param_map[id(v.data)] = k

    def reset_params(self):
        """Reset the param_map and _seen.

        Returns:
            AutoGradDot: self.
        """
        self.param_map = {}
        self._seen = set()
        self.inputs = self.inputs
        self.inputs_kwargs = self.inputs_kwargs
        self.module = self.module
        return self

    def get_tensor_name(self, tensor: Tensor, name: Optional[str] = None):
        """Get the name of the tensor.

        Args:
            tensor (Tensor): the tensor.
            name (Optional[str]): the name of the tensor. Defaults to None.

        Returns:
            str: the name of the tensor.
        """
        if not name:
            if id(tensor) in self.param_map:
                name = self.param_map[id(tensor)]
            elif hasattr(tensor, "name") and tensor.name is not None:
                name = tensor.name
            elif hasattr(tensor, "names") and tensor.names is not None and len(tensor.names) > 0:
                name = str(tensor.names)
            else:
                name = ''
        return '%s\n %s' % (name, size_to_str(tensor.size()))

    def add_tensor(self, tensor: Tensor, name: Optional[str] = None, _attributes=None, **kwargs):
        """Add a tensor to the graph.

        Args:
            tensor (Tensor): the tensor.
            name (Optional[str]): the name of the tensor. Defaults to None.
            _attributes (Optional[Dict[str, str]]): the attributes of the tensor. Defaults to None.
            **kwargs: the attributes of the dot.node function.

        Returns:
            AutoGradDot: self.
        """
        self._seen.add(tensor)
        self.dot.node(
            name=str(id(tensor)),
            label=self.get_tensor_name(tensor, name=name),
            _attributes=_attributes,
            **kwargs
        )
        return self

    def add_fn(self, fn: Any, _attributes=None, **kwargs):
        """Add a function to the graph.

        Args:
            fn (Any): the function.
            _attributes (Optional[Dict[str, str]]): the attributes of the function. Defaults to None.
            **kwargs: the attributes of the dot.node function.

        Returns:
            AutoGradDot: self.
        """
        self._seen.add(fn)
        self.dot.node(
            name=str(id(fn)),
            label=get_fn_name(fn, self.show_attrs, self.max_attr_chars),
            _attributes=_attributes,
            **kwargs
        )
        return self

    def add_edge(self, u: Any, v: Any, label: Optional[str] = None, _attributes=None, **kwargs):
        """Add an edge to the graph.

        Args:
            u (Any): tail node.
            v (Any): head node.
            label (Optional[str]): the label of the edge. Defaults to None.
            _attributes (Optional[Dict[str, str]]): the attributes of the edge. Defaults to None.
            **kwargs: the attributes of the dot.edge function.

        Returns:
            AutoGradDot: self.
        """
        self.dot.edge(
            tail_name=str(id(u)),
            head_name=str(id(v)),
            label=label,
            _attributes=_attributes,
            **kwargs
        )
        return self

    def add_seen(self, item: Any):
        """Add an item to the seen set.

        Args:
            item (Any): the item.

        Returns:
            AutoGradDot: self.
        """
        self._seen.add(item)
        return self

    def is_seen(self, item: Any) -> bool:
        """Check if the item is in the seen set.

        Args:
            item (Any): the item.

        Returns:
            bool: True if the item is in the seen set.
        """
        return item in self._seen


def add_grad_fn(grad_fn, autograd_dot: AutoGradDot):
    """Add a grad_fn to the graph.

    Args:
        grad_fn (Any): the Tensor.grad_fn.
        autograd_dot (AutoGradDot): the AutoGradDot object.
    """
    assert not torch.is_tensor(grad_fn)
    if autograd_dot.is_seen(grad_fn):
        return

    # add the node for this grad_fn
    autograd_dot.add_fn(grad_fn)

    if autograd_dot.show_saved:
        for attr in dir(grad_fn):
            if not attr.startswith(SAVED_PREFIX):
                continue
            val = getattr(grad_fn, attr)
            autograd_dot.add_seen(val)
            attr = attr[len(SAVED_PREFIX):]
            if torch.is_tensor(val):
                autograd_dot.add_edge(grad_fn, val, dir="none")
                autograd_dot.add_tensor(val, name=attr, fillcolor='orange')
            if isinstance(val, tuple):
                for i, t in enumerate(val):
                    if torch.is_tensor(t):
                        name = attr + '[%s]' % str(i)
                        autograd_dot.add_edge(grad_fn, t, dir="none")
                        autograd_dot.add_tensor(t, name=name, fillcolor='orange')

    if hasattr(grad_fn, 'variable'):
        # if grad_accumulator, add the node for `.variable`
        var = grad_fn.variable
        autograd_dot.add_tensor(var, fillcolor='lightblue')
        autograd_dot.add_edge(var, grad_fn)

    # recurse
    if hasattr(grad_fn, 'next_functions'):
        for u in grad_fn.next_functions:
            if u[0] is not None:
                autograd_dot.add_edge(u[0], grad_fn)
                add_grad_fn(u[0], autograd_dot=autograd_dot)

    # note: this used to show .saved_tensors in pytorch0.2, but stopped
    # working* as it was moved to ATen and Variable-Tensor merged
    # also note that this still works for custom autograd functions
    if hasattr(grad_fn, 'saved_tensors'):
        for t in grad_fn.saved_tensors:
            autograd_dot.add_edge(t, grad_fn)
            autograd_dot.add_tensor(t, fillcolor='orange')


def add_base_tensor(tensor: Tensor, autograd_dot: AutoGradDot, color: str = 'darkolivegreen1'):
    """Add a base tensor to the graph.

    Args:
        tensor (Tensor): the base tensor.
        autograd_dot (AutoGradDot): the AutoGradDot object.
        color (str): the color of the base tensor in graph. Defaults to 'darkolivegreen1'.
    """
    if autograd_dot.is_seen(tensor):
        return

    autograd_dot.add_tensor(tensor, fillcolor=color)

    if tensor.grad_fn:
        add_grad_fn(tensor.grad_fn, autograd_dot=autograd_dot)
        autograd_dot.add_edge(tensor.grad_fn, tensor)

    if tensor._is_view():
        add_base_tensor(tensor._base, autograd_dot=autograd_dot, color='darkolivegreen3')
        autograd_dot.add_edge(tensor._base, tensor, style="dotted")


def make_autograd_obj(
        module: nn.Module,
        *args: Tensor,
        additional_params: Optional[dict] = None,
        show_attrs: bool = True,
        show_saved: bool = True,
        max_attr_chars: int = 50,
        **kwargs: Tensor
) -> AutoGradDot:
    """Produces Graphviz representation of PyTorch autograd graph.

    If a node represents a backward function, it is gray. Otherwise, the node
    represents a tensor and is either blue, orange, or green:
     - Blue: reachable leaf tensors that requires grad (tensors whose `.grad`
         fields will be populated during `.backward()`)
     - Orange: saved tensors of custom autograd functions as well as those
         saved by built-in backward nodes
     - Green: tensor passed in as outputs
     - Dark green: if any output is a view, we represent its base tensor with
         a dark green node.

    Args:
        module (nn.Module): PyTorch model
        *args (Tensor): input to the model
        additional_params (dict): dict of additional params to label nodes with
        show_attrs (bool): whether to display non-tensor attributes of backward nodes
            (Requires PyTorch version >= 1.9)
        show_saved (bool): whether to display saved tensor nodes that are not by custom
            autograd functions. Saved tensor nodes for custom functions, if
            present, are always displayed. (Requires PyTorch version >= 1.9)
        max_attr_chars (int): if show_attrs is `True`, sets max number of characters
            to display for any given attribute.
        **kwargs (Tensor): input to the model

    Returns:
        AutoGradDot: graphviz representation of autograd graph
    """

    if LooseVersion(torch.__version__) < LooseVersion("1.9") and (show_attrs or show_saved):
        warnings.warn(
            "make_dot: showing grad_fn attributes and saved variables"
            " requires PyTorch version >= 1.9. (This does NOT apply to"
            " saved tensors saved by custom autograd functions.)"
        )

    autograd_dot = AutoGradDot()
    autograd_dot.module = module
    autograd_dot.inputs = args
    autograd_dot.inputs_kwargs = kwargs
    autograd_dot.reset_params()
    autograd_dot.show_attrs = show_attrs
    autograd_dot.show_saved = show_saved
    autograd_dot.max_attr_chars = max_attr_chars

    if additional_params is not None:
        autograd_dot.param_map.update(additional_params)

    # handle multiple outputs
    for v in autograd_dot.outputs:
        add_base_tensor(tensor=v, autograd_dot=autograd_dot)

    resize_graph(autograd_dot.dot)
    return autograd_dot


def join_scope_name(name: str, scope: Dict[str, str]) -> str:
    return '/'.join([scope[name], name])


def parse_trace_graph(graph) -> List[Node]:
    scope = {}
    for n in graph.nodes():
        inputs = [i.uniqueName() for i in n.inputs()]
        for i in range(1, len(inputs)):
            scope[inputs[i]] = n.scopeName()

        uname = next(n.outputs()).uniqueName()
        assert n.scopeName() != '', '{} has empty scope name'.format(n)
        scope[uname] = n.scopeName()

    scope['0'] = 'input'

    nodes = []
    for n in graph.nodes():
        attrs = {k: n[k] for k in n.attributeNames()}
        attrs = str(attrs).replace("'", ' ')
        inputs = [join_scope_name(i.uniqueName(), scope) for i in n.inputs()]
        uname = next(n.outputs()).uniqueName()

        nodes.append(Node(
            name=join_scope_name(uname, scope),
            op=n.kind(),
            inputs=inputs,
            attr=attrs
        ))

    for n in graph.inputs():
        uname = n.uniqueName()
        if uname not in scope.keys():
            scope[uname] = 'unused'

        nodes.append(Node(
            name=join_scope_name(uname, scope),
            op='Parameter',
            inputs=[],
            attr=str(n.type())
        ))

    return nodes


def get_autograd_dot_from_trace(trace) -> Digraph:
    """ Produces graphs of torch.jit.trace outputs

    Example:
    >>> trace, = torch.jit.trace(model, args=(x,))
    >>> dot = get_autograd_dot_from_trace(trace)

    Args:
        trace (torch.jit.trace): the trace object to visualize.

    Returns:
        graphviz.Digraph: the resulting graph.
    """

    try:
        from graphviz import Digraph
    except ImportError as e:
        raise ImportError("requires graphviz: https://pygraphviz.github.io/") from e

    # from tensorboardX
    if LooseVersion(torch.__version__) >= LooseVersion("0.4.1"):
        torch.onnx._optimize_trace(trace, torch._C._onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    elif LooseVersion(torch.__version__) >= LooseVersion("0.4"):
        torch.onnx._optimize_trace(trace, False)
    else:
        torch.onnx._optimize_trace(trace)
    graph = trace.graph()
    list_of_nodes = parse_trace_graph(graph)

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')

    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

    for node in list_of_nodes:
        dot.node(node.name, label=node.name.replace('/', '\n'))
        if node.inputs:
            for inp in node.inputs:
                dot.edge(inp, node.name)

    resize_graph(dot)

    return dot


def get_autograd_obj(module: nn.Module, *args, **kwargs) -> AutoGradDot:
    """Produces Graphviz representation of PyTorch autograd graph.

    Args:
        module: PyTorch model
        *args: input to the make_autograd_obj
        **kwargs: input to the make_autograd_obj

    Returns:
        AutoGradDot: graphviz representation of autograd graph
    """
    assert isinstance(module, nn.Module)
    new_args = []
    new_kwargs = {}
    device = is_cpu_cuda.get_module_device(module)

    for i in args:
        assert isinstance(i, Tensor)
        i = i.to(device)
        i = i.detach()
        i = i.requires_grad_(True)
        new_args.append(i)

    for k, v in kwargs.items():
        assert isinstance(v, Tensor)
        v = v.to(device)
        v = v.detach()
        v = v.requires_grad_(True)
        new_kwargs[k] = v

    training_status = module.training
    use_autograd_graph = False
    if isinstance(module, Layer):
        use_autograd_graph = module.use_autograd_graph
        module.use_autograd_graph = True
    module.eval()

    autograd_dot = make_autograd_obj(module, *new_args, **new_kwargs)

    module.train(training_status)
    if isinstance(module, Layer):
        module.use_autograd_graph = use_autograd_graph
    return autograd_dot


def get_autograd_dot(module: nn.Module, *args, **kwargs) -> Digraph:
    """Produces Graphviz representation of PyTorch autograd graph.

    Args:
        module: PyTorch model
        *args: input to the make_autograd_obj
        **kwargs: input to the make_autograd_obj

    Returns:
        Digraph: graphviz representation of autograd graph
    """
    return get_autograd_obj(module, *args, **kwargs).dot


def save_autograd_graph(filename, module: nn.Module, *args, **kwargs) -> None:
    """Save Graphviz representation of PyTorch autograd graph.

    Args:
        filename: filename to save the graph to
        module: PyTorch model
        *args: input to the make_autograd_obj
        **kwargs: input to the make_autograd_obj
    """

    get_autograd_dot(module, *args, **kwargs).render(filename, format="svg", cleanup=True)


def save_autograd_graph_from_trace(filename, trace) -> None:
    """Save Graphviz representation of PyTorch autograd graph from trace.

    Args:
        filename: filename to save the graph to
        trace (torch.jit.trace): the trace object to visualize.
    """

    get_autograd_dot_from_trace(trace).render(filename, format="svg", cleanup=True)
