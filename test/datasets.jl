# funs to test dataset availibility
#
function test_mit_nsr_download()
    nsr = dataset_mit_nsr("16265"; force=true)

    return length(nsr) == 1
end


function test_dataset_mit_nsr_saved()
    nsr = dataset_mit_nsr()

    return length(nsr) == 18
end


function test_dataset_mnist()
    mnist = dataset_mnist()

    return length(mnist) == 4
end

function test_dataset_iris()
    iris = dataset_mit_nsr()

    return DataFrames.nrow(iris) == 150
end