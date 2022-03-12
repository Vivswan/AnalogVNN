from nn.utils.make_dot import make_dot


def save_graph(filename, output, named_parameters=None):
    if named_parameters is not None:
        named_parameters = dict(named_parameters)
    make_dot(output, params=named_parameters).render(filename, format="svg", cleanup=True)
