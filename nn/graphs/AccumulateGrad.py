import torch

from nn.graphs.ArgsKwargs import ArgsKwargs


class AccumulateGrad:
    def __init__(self, module):
        self.locations = {}
        self.module = module

    def __repr__(self):
        # return f"AccumulateGrad"
        return f"AccumulateGrad({self.module})"

    def grad(self, grad_outputs_args_kwargs: ArgsKwargs, forward_input_output_graph) -> ArgsKwargs:
        grad_inputs_args = {}
        grad_inputs_kwargs = {}
        for key, grad_output in grad_outputs_args_kwargs.kwargs.items():
            location = self.locations[key]
            forward_out_arg = location['in_arg']
            forward_out_kwarg = location['in_kwarg']
            forward_in_arg = location['out_arg']
            forward_in_kwarg = location['out_kwarg']
            # print(out_kwarg, out_arg, value)

            # 0 - not allowed

            # 4
            if forward_out_arg is True and isinstance(forward_in_arg, int) and not isinstance(forward_in_arg, bool):
                forward_inputs = forward_input_output_graph[location["from"]].inputs.args
                forward_outputs = forward_input_output_graph[self.module].outputs.args
                forward_out_arg = forward_inputs.index(forward_outputs[forward_in_arg])
                grad_output = grad_output[forward_out_arg]

            # 7
            if forward_out_arg is True and isinstance(forward_in_kwarg, str):
                forward_inputs = forward_input_output_graph[location["from"]].inputs.args
                forward_outputs = forward_input_output_graph[self.module].outputs.kwargs
                forward_out_arg = forward_inputs.index(forward_outputs[forward_in_kwarg])
                grad_output = grad_output[forward_out_arg]

            # 1
            if forward_out_arg is True and forward_in_arg is True:
                forward_inputs = forward_input_output_graph[location["from"]].inputs.args
                forward_outputs = forward_input_output_graph[self.module].outputs.args
                for i in range(len(forward_inputs)):
                    if forward_inputs[i] not in forward_outputs:
                        continue

                    value_index = forward_outputs.index(forward_inputs[i])
                    if value_index not in grad_inputs_args:
                        grad_inputs_args[value_index] = torch.zeros_like(grad_output[i])
                    grad_inputs_args[value_index] += grad_output[i]
                continue

            # 2
            if forward_out_arg is True and forward_in_kwarg is True:
                forward_inputs = forward_input_output_graph[location["from"]].inputs.args
                forward_outputs = forward_input_output_graph[self.module].outputs.kwargs
                for i in forward_outputs:
                    value_index = forward_inputs.index(forward_outputs[i])

                    if i not in grad_inputs_kwargs:
                        grad_inputs_kwargs[i] = torch.zeros_like(grad_output[value_index])
                    grad_inputs_kwargs[i] += grad_output[value_index]
                continue

            # 3
            if forward_out_kwarg is True and forward_in_kwarg is True:
                for i in grad_output:
                    if i not in grad_inputs_kwargs:
                        grad_inputs_kwargs[i] = torch.zeros_like(grad_output[i])

                    grad_inputs_kwargs[i] += grad_output[i]
                continue

            # 8 & 9
            if forward_in_kwarg is not None and isinstance(forward_in_kwarg, str):
                if forward_in_kwarg not in grad_inputs_kwargs:
                    grad_inputs_kwargs[forward_in_kwarg] = torch.zeros_like(grad_output)

                grad_inputs_kwargs[forward_in_kwarg] += grad_output
                continue

            # 5 & 6
            if forward_in_arg is not None and isinstance(forward_in_arg, int) and not isinstance(forward_in_arg, bool):
                if forward_in_arg not in grad_inputs_args:
                    grad_inputs_args[forward_in_arg] = torch.zeros_like(grad_output)
                grad_inputs_args[forward_in_arg] += grad_output
                continue

            raise NotImplementedError("WTF!Why!")

        return ArgsKwargs(
            args=[grad_inputs_args[i] for i in sorted(list(grad_inputs_args.keys()))],
            kwargs=grad_inputs_kwargs
        )
