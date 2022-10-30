*********************************************
Linear3 Photonic Analog Neural Network
*********************************************

Sample code for 3 layered linear photonic analog neural network with 4-bit precision, 0.2 leakage and clamp normlization:

.. code-block:: python
    import torch.backends.cudnn
    import torchvision
    from torch import optim, nn

    from dataloaders.load_vision_dataset import load_vision_dataset
    from nn.activations.Gaussian import GeLU
    from nn.layers.Linear import Linear
    from nn.layers.functionals.BackwardWrapper import BackwardWrapper
    from nn.layers.functionals.Normalize import Clamp
    from nn.layers.functionals.ReducePrecision import ReducePrecision
    from nn.layers.noise.GaussianNoise import GaussianNoise
    from nn.modules.FullSequential import FullSequential
    from nn.modules.Sequential import Sequential
    from nn.optimizer.PseudoOptimizer import PseudoOptimizer
    from nn.utils.is_cpu_cuda import is_cpu_cuda


    def cross_entropy_accuracy(output, target) -> float:
        """ Cross Entropy accuracy function

        Args:
            output (torch.Tensor): output of the model from passing inputs
            target (torch.Tensor): correct labels for the inputs

        Returns:
            float: accuracy from 0 to 1
        """
        _, preds = torch.max(output.data, 1)
        correct = (preds == target).sum().item()
        return correct / len(output)


    class LinearModel(FullSequential):
        def __init__(self, activation_class, norm_class, precision_class, precision, noise_class, leakage):
            """ Linear Model with 3 dense layers

            Args:
                activation_class: Activation Class
                norm_class: Normalization Class
                precision_class: Precision Class (ReducePrecision or StochasticReducePrecision)
                precision (int): precision of the weights and biases
                noise_class: Noise Class
                leakage (float): leakage is the probability that a reduced precision digital value (e.g., “1011”) will
                acquire a different digital value (e.g., “1010” or “1100”) after passing through the noise layer
                (i.e., the probability that the digital values transmitted and detected are different after passing through
                the analog channel).
            """
            super(LinearModel, self).__init__()

            self.activation_class = activation_class
            self.norm_class = norm_class
            self.precision_class = precision_class
            self.precision = precision
            self.noise_class = noise_class
            self.leakage = leakage

            self.all_layers = []
            self.all_layers.append(BackwardWrapper(nn.Flatten(start_dim=1)))
            self.add_layer(Linear(in_features=28 * 28, out_features=256))
            self.add_layer(Linear(in_features=256, out_features=128))
            self.add_layer(Linear(in_features=128, out_features=10))

            self.add_sequence(*self.all_layers)

        def add_layer(self, layer):
            """ To add the analog layer

            Args:
                layer (BaseLayer): digital layer module
            """
            self.all_layers.append(self.norm_class())
            self.all_layers.append(self.precision_class(precision=self.precision))
            self.all_layers.append(self.noise_class(leakage=self.leakage, precision=self.precision))
            self.all_layers.append(layer)
            self.all_layers.append(self.noise_class(leakage=self.leakage, precision=self.precision))
            self.all_layers.append(self.norm_class())
            self.all_layers.append(self.precision_class(precision=self.precision))
            self.all_layers.append(self.activation_class())
            self.activation_class.initialise_(layer.weight)


    class WeightModel(Sequential):
        def __init__(self, norm_class, precision_class, precision, noise_class, leakage):
            """

            Args:
                norm_class: Normalization Class
                precision_class: Precision Class (ReducePrecision or StochasticReducePrecision)
                precision (int): precision of the weights and biases
                noise_class: Noise Class
                leakage (float): leakage is the probability that a reduced precision digital value (e.g., “1011”) will
                acquire a different digital value (e.g., “1010” or “1100”) after passing through the noise layer
                (i.e., the probability that the digital values transmitted and detected are different after passing through
                the analog channel).
            """
            super(WeightModel, self).__init__()
            self.all_layers = []

            self.all_layers.append(norm_class())
            self.all_layers.append(precision_class(precision=precision))
            self.all_layers.append(noise_class(leakage=leakage, precision=precision))

            self.eval()
            self.add_sequence(*self.all_layers)


    def run_linear3_model():
        torch.backends.cudnn.benchmark = True
        device, is_cuda = is_cpu_cuda.is_using_cuda()
        print(f"Device: {device}")
        print()

        # Loading Data
        print(f"Loading Data...")
        train_loader, test_loader, input_shape, classes = load_vision_dataset(
            dataset=torchvision.datasets.MNIST,
            path="_data/",
            batch_size=128,
            is_cuda=is_cuda
        )

        # Creating Models
        print(f"Creating Models...")
        nn_model = LinearModel(
            activation_class=GeLU,
            norm_class=Clamp,
            precision_class=ReducePrecision,
            precision=2 ** 4,
            noise_class=GaussianNoise,
            leakage=0.2
        )
        weight_model = WeightModel(
            norm_class=Clamp,
            precision_class=ReducePrecision,
            precision=2 ** 4,
            noise_class=GaussianNoise,
            leakage=0.2
        )

        # Setting Model Parameters
        nn_model.loss_fn = nn.CrossEntropyLoss()
        nn_model.accuracy_fn = cross_entropy_accuracy

        nn_model.compile(device=device)
        nn_model.to(device=device)
        weight_model.to(device=device)

        PseudoOptimizer.parameter_type.convert_model(nn_model, transform=weight_model)
        nn_model.optimizer = PseudoOptimizer(
            optimizer_cls=optim.Adam,
            params=nn_model.parameters(),
        )

        # Training
        print(f"Starting Training...")
        for epoch in range(10):
            train_loss, train_accuracy = nn_model.train_on(train_loader, epoch=epoch)
            test_loss, test_accuracy = nn_model.test_on(test_loader, epoch=epoch)

            str_epoch = str(epoch + 1).zfill(1)
            print_str = f'({str_epoch})' \
                        f' Training Loss: {train_loss:.4f},' \
                        f' Training Accuracy: {100. * train_accuracy:.0f}%,' \
                        f' Testing Loss: {test_loss:.4f},' \
                        f' Testing Accuracy: {100. * test_accuracy:.0f}%\n'
            print(print_str)
        print("Run Completed Successfully...")


    if __name__ == '__main__':
        run_linear3_model()
