********************
Extra Analog Classes
********************
Some extra layers which can be found in AnalogVNN are as follows:

Reduce Precision
================
Reduce Precision classes are used to reduce precision of an input to some given precision level

ReducePrecision
_________________
Reduce Precision uses the following function to reduce precision of the input value

.. math::

    RP(x) = sign(x * p) * max(\left\lfloor \left| x * p \right| \right\rfloor, \left\lceil \left| x * p \right| - d \right\rceil) * \frac{1}{p}

where:

x is the original number in full precision

p is the analog precision of the input signal, output signal, or weights (p ∈ Natural Numbers, number\ of\ bits=\ {log}_2\left(p+1\right))

d is the divide parameter (0 ≤ d ≤ 1, default value = 0.5) which determines whether x is rounded to a discrete level higher or lower than the original value


StochasticReducePrecision
_________________
Coming Soon...

Normalization
=============
Coming Soon...

LPNorm
______
Coming Soon...

LPNormW
_______
Coming Soon...

LPNormWM
_______
Coming Soon...

Noise
=====
Coming Soon...

GaussianNoise
_____________
Coming Soon...

LaplacianNoise
______________
Coming Soon...

PoissonNoise
____________
Coming Soon...

UniformNoise
____________
Coming Soon...

