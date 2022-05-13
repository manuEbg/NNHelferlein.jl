# NNHelferlein.jl
Collection of little helpers to simplify various Machine Learning tasks
(esp. building neural networks with Knet).

The German word *Helferlein* means something like *little helper*;
please pronounce it like `hell-fur-line`.

The package follows mainly the Knet-style; i.e. all networks can be trained with the Knet-iterators, all layers can be used together with Knet-style quickly-self-written layers, all Knet-networks can be trained with tb_train(), all data providers can be used together, ...

The high-level API makes its possible to define and train neural networks in less
lines of code and even more intuitively as with frameworks like TensorFlow or 
PyTorch.

See documentation and examples for a first intro.

<!---
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://KnetML.github.io/NNHelferlein.jl/stable)
--->
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://KnetML.github.io/NNHelferlein.jl/dev)
<!--
CI badge
[![Build Status](https://travis-ci.org/KnetML/NNHelferlein.jl.svg?branch=main)](https://travis-ci.org/KnetML/NNHelferlein.jl)
-->
[![Tests](https://github.com/KnetML/NNHelferlein.jl/actions/workflows/run_tests.yml/badge.svg)](https://github.com/KnetML/NNHelferlein.jl/actions/workflows/run_tests.yml) [![codecov](https://codecov.io/gh/KnetML/NNHelferlein.jl/branch/main/graph/badge.svg?token=9R12TMSKP1)](https://codecov.io/gh/KnetML/NNHelferlein.jl)


# Installation

The package is not yet released! Please install the legacy project at 
<https://github.com/andreasdominik/NNHelferlein.jl>.

```Julia
using Pkg
Pkg.add(url="https://github.com/andreasdominik/NNHelferlein.jl.git")
```

<!---
Due to a backwards incompatibility with the dependency `AutoGrad.jl`, it is
currently necessary to manually install the latest version of AutoGrad.jl instead
of the released version 1.2.4 to be used with NNHelferlein:

```Julia
using Pkg
Pkg.add(url="https://github.com/andreasdominik/NNHelferlein.jl.git")
Pgk.add(url="https://github.com/denizyuret/AutoGrad.jl.git")
```
--->


# Caveat:
Please be aware that the package is still in development and
not yet completely tested. You may already use it on own risk.

While reading this, I must add: the package is *almost* ready with an
not-so-bad test coverage. If you see the tests passing in the moment, 
it may be save to use the helpers.

As soon as dev. and tests are completed the package will be
registered soon (i.e. already in a few days).
