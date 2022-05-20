# NNHelferlein.jl
Collection of little helpers to simplify various Machine Learning tasks
(esp. building neural networks with Knet).

The German word *Helferlein* means something like *little helper*;
please pronounce it like `hell-fur-line`.

The package follows mainly the Knet-style; i.e. all networks can be trained with the Knet-iterators, all layers can be used together with Knet-style quickly-self-written layers, all Knet-networks can be trained with tb_train!(), all data providers can be used together, ...

The high-level API makes its possible to define and train neural networks in less
lines of code and even more intuitively as with high-level (Python-) frameworks.

See documentation and examples for a first intro.

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://KnetML.github.io/NNHelferlein.jl/stable)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://KnetML.github.io/NNHelferlein.jl/dev)
[![Tests](https://github.com/KnetML/NNHelferlein.jl/actions/workflows/run_tests.yml/badge.svg)](https://github.com/KnetML/NNHelferlein.jl/actions/workflows/run_tests.yml) [![codecov](https://codecov.io/gh/KnetML/NNHelferlein.jl/branch/main/graph/badge.svg?token=9R12TMSKP1)](https://codecov.io/gh/KnetML/NNHelferlein.jl)


# Installation

The package can be installed with the package manager as:

```JuliaREPL
] add NNHelferlein
```
or
```JuliaREPL
using Pkg
Pkg.add("NNHelferlein")
```
