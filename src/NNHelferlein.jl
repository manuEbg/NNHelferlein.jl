module NNHelferlein

import Base.iterate
import Base.length
import Knet
using Knet: KnetArray, Param
import CUDA
import CSV
import OdsIO
import DataFrames
import Random
import Images
import Augmentor
import MLDataUtils


include("nets.jl")
include("layers.jl")
include("images.jl")
include("dataframes.jl")


export NeuNet, Classifier, Regressor,           # chains
       ImageLoader, iterate,
       Dense, Conv, Pool, Flat, PyFlat,         # layers
       Embed,
       dataframe_read, dataframe_split,         # import data
       dataframe_minibatches,
       mk_image_minibatch

end # module
