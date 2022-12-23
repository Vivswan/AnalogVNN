from dataclasses import dataclass

from nn.graphs.ArgsKwargs import ArgsKwargs


@dataclass
class InputOutput:
    def __init__(self, inputs=None, outputs=None):
        self.inputs: ArgsKwargs = inputs
        self.outputs: ArgsKwargs = outputs

    def __repr__(self):
        return f"InputOutput(inputs={self.inputs}, outputs={self.outputs})"
