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

    p is the analog precision of the input signal, output signal, or weights (p ∈ Natural Numbers, number of bits= log2(p+1))

    d is the divide parameter (0 ≤ d ≤ 1, default value = 0.5) which determines whether x is rounded to a discrete level higher or lower than the original value


StochasticReducePrecision
_________________
Reduce Precision uses the following probabilistic function to reduce precision of the input value

.. math::

    SRP(x) = sign(x*p) * f(\left| x*p \right|) * \frac{1}{p}

    f(x)  = \left\{ \begin{array}{cl}
    \left\lfloor x \right\rfloor & : \ r \le 1 - \left| \left\lfloor x \right\rfloor - x \right| \\
    \left\lceil x \right\rceil & : otherwise
    \end{array} \right.

where:
	r is a uniformly distributed random number between 0 and 1

    p is the analog precision (p ∈ Natural Numbers, number of bits= log2(p+1))

    f(x) is the stochastic rounding function

Normalization
=============

LPNorm
______
.. math::

    L^pNorm(x) = \left[ {x}_{ij..k} \to \frac{{x}_{ij..k}}{\sqrt[p]{\sum_{j..k}^{} \left| {x}_{ij..k} \right|^p}} \right]

    L^pNormM(x) = \frac{L^pNorm(x)}{max(\left| L^pNorm(x) \right|))}

where:
    x is the input weight matrix,

    i, j ... k are indexes of the matrix,

    p is a positive integer.

LPNormW
_______
.. math::

    L^pNormW(x) = \frac{x}{\left\| x \right\|_p} = \frac{x}{\sqrt[p]{\sum_{}^{} \left| x \right|^p}}

    L^pNormWM(x) = \frac{L^pNormW(x)}{max(\left| L^pNormW(x) \right|))}


where:
    x is the input weight matrix,

    p is a positive integer.

Clamp
_______
.. math::

    Clamp_{pq}(x) = \left\{ \begin{array}{cl}
    q & : \ q \lt x \\
    x & : \ p \le x \le q \\
    p & : \ p \gt x
    \end{array} \right.

where:
	p, q ∈ ℜ (p ≤ q, Default value for photonics p = −1 and q = 1)

Noise
=====

Leakage or Error Probability
_____________
We have defined an information loss parameter, "error probability" or "EP," as the probability that a reduced precision 
digital value (e.g., "1011") will acquire a different digital value (e.g., "1010" or "1100") after passing through the
noise layer (i.e., the probability that the digital values transmitted and detected are different after passing through
the analog channel). This is a similar concept to the bit error ratio (BER) used in digital communications, but for numbers
with multiple bits of resolution. While SNR (signal-to-noise ratio) is inversely proportional to \sigma, the standard
deviation of the signal noise, EP is indirectly proportional to σ. However, we choose EP since it provides a more intuitive
understanding of the effect of noise in an analog system from a digital perspective. It is also similar to the rate
parameter used in PyTorch’s Dropout Layer [23], though different in function. EP is defined as follows:

.. math::



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

