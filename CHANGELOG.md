# Changelog

## 1.0.0

* Public release

## 1.0.1 (Patches for Pytorch 2.0.0)

* added `grad.setter` to `PseudoParameterModule` class

## 1.0.2

* Bugfix: removed  `graph` from `Layer` class
  * `graph` was causing issues with nested `Model` objects
  * Now `_use_autograd_graph` is directly set while compiling the `Model` object

## 1.0.3

* Added support for no loss function in `Model` class
  * If no loss function is provided, the `Model` object will use outputs for gradient computation
* Added support for multiple loss outputs from loss function