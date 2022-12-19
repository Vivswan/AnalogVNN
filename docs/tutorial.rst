********
Tutorial
********

Custom Forward and backward graphs

To convert a digital model to its analog counterpart the following steps needs to be followed:

#. Adding the analog layers to the digital model. For example, to create the Photonic Linear Layer with Reduce Precision, Normalization and Noise:
    #. Create the model similar to how you would create a digital model but using FullSequential as superclass
        .. code-block:: python

            class LinearModel(FullSequential):
                def __init__(self, activation_class, norm_class, precision_class, precision, noise_class, leakage):
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

       Note: "add_sequence" is used to create and set forward and backward graphs in AnalogVNN, more information in :doc:`inner_workings`

    #. To add the Reduce Precision, Normalization, and Noise before and after the main Linear layer, we can modify the "add_layer" function
        .. code-block:: python

            def add_layer(self, layer):
                self.all_layers.append(self.norm_class())
                self.all_layers.append(self.precision_class(precision=self.precision))
                self.all_layers.append(self.noise_class(leakage=self.leakage, precision=self.precision))
                self.all_layers.append(layer)
                self.all_layers.append(self.noise_class(leakage=self.leakage, precision=self.precision))
                self.all_layers.append(self.norm_class())
                self.all_layers.append(self.precision_class(precision=self.precision))
                self.all_layers.append(self.activation_class())
                self.activation_class.initialise_(layer.weight)

#. Creating an Analog Parameters Model for analog parameters (analog weights and biases)
    .. code-block:: python

        class WeightModel(FullSequential):
            def __init__(self, norm_class, precision_class, precision, noise_class, leakage):
                super(WeightModel, self).__init__()
                self.all_layers = []

                self.all_layers.append(norm_class())
                self.all_layers.append(precision_class(precision=precision))
                self.all_layers.append(noise_class(leakage=leakage, precision=precision))

                self.eval()
                self.add_sequence(*self.all_layers)

    Note: Since the "WeightModel" will only be used for converting the data to analog data to be used in the main "LinearModel", we can use "eval()" to make sure the "WeightModel" is never been trained

#. Simply getting data and setting up the model as we will normally do in PyTorch with some minor changes for automatic evaluations
    .. code-block:: python

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
        nn_model.loss_function = nn.CrossEntropyLoss()
        nn_model.accuracy_function = cross_entropy_accuracy
        nn_model.compile(device=device)
        weight_model.compile(device=device)
        nn_model.to(device=device)
        weight_model.to(device=device)


#. Using Analog Parameters Model to, convert digital parameters to analog parameters
    .. code-block:: python

        PseudoParameter.convert_model(nn_model, transform=weight_model)

#. Converting the digital optimizer to analog optimizer
    .. code-block:: python

        nn_model.optimizer = PseudoOptimizer(
            optimizer_cls=optim.Adam,
            params=nn_model.parameters(),
        )

#. Then you are good to go, and train and test the model
    .. code-block:: python

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

Full Sample code for this process can be found at :doc:`sample_code`
