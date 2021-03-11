# using MLDatasets: MNIST
using Base.Iterators
import Pkg; Pkg.add("Augmentor"); using Augmentor
using DataFrames

function test_image_loader()
    augm = CropSize(28,28)
    trn, vld = mk_image_minibatch("../data/flowers",
                4; split=true, fr=0.2,
                balanced=false, shuffle=true,
                train=true,
                aug_pipl=augm, pre_proc=nothing)

    return size(first(trn)[1]) == (28,28,3,4)
end

function test_lenet()

    augm = CropSize(28,28)
    trn, vld = mk_image_minibatch("../data/flowers",
                4; split=true, fr=0.2,
                balanced=false, shuffle=true,
                train=true,
                aug_pipl=augm, pre_proc=nothing)

    lenet = Classifier(Conv(5,5,3,20),
                    Pool(),
                    Conv(5,5,20,50),
                    BatchNorm(;trainable=true, channels = 50),
                    Pool(),
                    Flat(),
                    Dense(800,512),
                    Linear(512, 512, actf=relu),
                    Predictions(512,3)
            )

    mdl = tb_train!(lenet, Adam, trn, vld, epochs=1,
            acc_fun=accuracy,
            eval_size=0.25, eval_freq=2, mb_loss_freq=100,
            tb_name="test_run", tb_text="NNHelferlein example")

    acc = accuracy(mdl, data=vld)
    return acc isa Real && acc <= 1.0
end


function test_mlp()

        trn = DataFrame(x1=randn(16), x2=randn(16),
                        x3=randn(16), x4=randn(16),
                        x5=randn(16), x6=randn(16),
                        x7=randn(16), x8=randn(16),
                        y=rand(1:4, 16))

        mb = dataframe_minibatches(trn, size=4, regression=true)

        mlp = Regressor(Dense(8,8, actf=relu),
                         Dense(8,8),
                         Predictions(8,1))

        mdl = tb_train!(mlp, Adam, trn, epochs=1, acc_fun=nothing)
        return acc isa Real && acc <= 1.0
end
