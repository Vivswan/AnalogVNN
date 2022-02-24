from collections import OrderedDict

from nn.modules.Sequential import Sequential


class FullSequential(Sequential):
    def add_sequence(self, *args):
        if len(args) == 0:
            return None

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            self.backward.add_relation(*([self.backward.OUTPUT] + list(reversed(args[0].values()))))
            for key, module in args[0].items():
                if module == self.backward.STOP:
                    continue
                self._add_run_module(key, module)
        else:
            self.backward.add_relation(*([self.backward.OUTPUT] + list(reversed(list(args)))))
            for idx, module in enumerate(args):
                if module == self.backward.STOP:
                    continue
                self._add_run_module(str(idx), module)

    def set_full_sequential_relation(self, *args):
        self.add_sequence(*args)
