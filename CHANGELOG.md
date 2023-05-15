# Changelog

## 1.0.5 (Patches for Pytorch 2.0.1)

* Removed unnecessary `PseudoParameter.grad` property.
* Patch for Pytorch 2.0.1, add filtering inputs in `BackwardGraph._calculate_gradients`.

## 1.0.4

* Combined `PseudoParameter` and `PseudoParameterModule` for better visibility
  * BugFix: fixed save and load of state_dict of `PseudoParameter` and transformation module
* Removed redundant class `analogvnn.parameter.Parameter`

## 1.0.3

* Added support for no loss function in `Model` class.
  * If no loss function is provided, the `Model` object will use outputs for gradient computation.
* Added support for multiple loss outputs from loss function.

## 1.0.2

* Bugfix: removed  `graph` from `Layer` class.
  * `graph` was causing issues with nested `Model` objects.
  * Now `_use_autograd_graph` is directly set while compiling the `Model` object.

## 1.0.1 (Patches for Pytorch 2.0.0)

* added `grad.setter` to `PseudoParameterModule` class.

## 1.0.0

* Public release.
