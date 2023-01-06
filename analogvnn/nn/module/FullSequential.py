from analogvnn.nn.module.Sequential import Sequential

__all__ = ['FullSequential']


class FullSequential(Sequential):
    def compile(self, device=None, layer_data=True):
        arr = [self.graphs.INPUT, *list(self._runtime_module_list.values()), self.graphs.OUTPUT]
        self.graphs.forward_graph.add_connection(*arr)
        self.graphs.backward_graph.add_connection(*reversed(arr))
        return super().compile(device, layer_data)
