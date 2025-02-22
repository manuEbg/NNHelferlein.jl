# Data loader and preprocessing funs for dataframes:
#
# (c) A. Dominik, 2020

"""
    dataframe_read(fname; o...)

Read a data table from an CSV-file with one sample
per row and return a DataFrame with the data.
(ODS-support is removed because of PyCall compatibility issues
of the OdsIO package).

All keyword arguments accepted by CSV.File() can be used.
"""
function dataframe_read(fname; o...)

    if occursin(r".*\.ods$", fname)
        println("Reading ODS-files is no longer supported!")
        return nothing
        # return readODS(fname)
    elseif occursin(r".*\.csv$", fname)
        return readCSV(fname; o...)
    else
        println("Error: unknown file format!")
        println("Please support csv file")
        return nothing
    end
end

# function readODS(fname)
#
#     printf("Reading data from ODS: $fname")
#     return OdsIO.ods_read(fname, retType="DataFrame")
# end

function readCSV(fname; o...)

    println("Reading data from CSV: $fname")
    return DataFrames.DataFrame(CSV.File(fname; o...))
end



"""
    dataframe_minibatch(data::DataFrames.DataFrame; size=256, ignore=[], teaching="y", 
                          verbose=1, o...)
    dataframe_minibatches()

Make Knet-conform minibatches of type `Knet.data` from a dataframe
with one sample per row.

`dataframe_minibatches()` is an alieas kept for backward compatibility.

### Arguments:
+ `ignore`: defines a list of column names to be ignored
+ `teaching="y"`: defines the column name with teaching input. Default is "y".
                `teaching` is handled differently, depending on its type:
                If `Int`, the teaching input is interpreted as
                class IDs and directly used for training (this assumes that
                the values range from 1..n). If type is a String, values are
                interpreted as class labels and converted to numeric class IDs
                by calling `mk_class_ids()`. The list of valid lables and their
                order can be created by calling `mk_class_ids(data.y)[2]`.
                If teaching is a scalar value, regression context is assumed,
                and the value is used unchanged for training.
+ `verbose=1`: if > 0, a summary of how the dataframe is used is echoed.
+ other keyword arguments: all keyword arguments accepted by
                `Knet.minibatch()` may be used.

Allowed column definitions for `ignore` and `teaching` include names (as Strings),
column names (as Symbols) or column indices (as Integer values).
"""
function dataframe_minibatch(data; size=16, ignore=[], teaching="y", 
                               verbose=1, o...)

    if !(ignore isa(AbstractArray))
        ignore = [ignore]
    end
    if eltype(ignore) <: Int
        ignore = names(data)[ignore]
    elseif eltype(ignore) <: Symbol
        ignore = String.(ignore)
    end
    if !isnothing(teaching)
        if teaching isa Int
            teaching = names(data)[teaching]
        elseif teaching isa Symbol
            teaching = String(teaching)
        end
        push!(ignore, teaching)
    end
    cols = filter(c->!(c in ignore), names(data))
    
    if verbose > 0
        println("making minibatches from DataFrame:")
        println("... number of records used:  $(Base.size(data,1))")
        println("... teaching input y is:     $teaching")
        println("... number of columns used:  $(length(cols))")

        print("... data columns:            ")
        if length(cols) <= 8
            println("$cols")
        else
            println("[$(cols[1]), $(cols[2]) ... $(cols[end-1]), $(cols[end])]")
        end
    end


    x = convert2KnetArray(data[!,cols])
    x = permutedims(x)

    if isnothing(teaching)
        if verbose > 0
            println("... no teaching input specified!")
        end

        return Knet.minibatch(x, size; o...)
    else
        # care for type of teaching column:
        #
        t_type = eltype(data[!,teaching])
        if t_type <: AbstractString       # make class_ids from labels
            teach = mk_class_ids(data[!,teaching])[1]
            y = permutedims(Array{UInt8}(teach))

        elseif t_type <: Integer          # take values as class_ids
            y = permutedims(Array{UInt8}(data[!,teaching]))

        elseif t_type <: Real             # use as is -> probably regression
            y = permutedims(convert2KnetArray(data[!,teaching]))

        else
            println("Don't know how to handle teaching input of type $t_type")
            return nothing
        end

        if verbose > 0
            if t_type <: Integer
                println("... number of classes:       $(length(unique(data[!,teaching])))")
            elseif t_type <: Real 
                println("... y is scalar in range     $(minimum(data[!,teaching])) - $(maximum(data[!,teaching]))")
            else    
                println("... number of classes:       $(length(unique(data[!,teaching])))")
            end
        end

        return Knet.minibatch(x, y, size; o...)
    end
end

dataframe_minibatches = dataframe_minibatch




"""
    function mk_class_ids(labels)

Take a list with n class labels for n instances and return a list of
n class-IDs (of type Int) and an array of lables with the array index
of each label corresponds its ID.


### Arguments:
+ `labels`: List of labels (typically Strings)

### Result values:
+ array of class-IDs in the same order as the input
+ array of unique class-IDs ordered by their ID.


### Examples:
```
julia> labels = ["blue", "red", "red", "red", "green", "blue", "blue"]
7-element Array{String,1}:
 "blue"
 "red"
 "red"
 "red"
 "green"
 "blue"
 "blue"

julia> mk_class_ids(labels)[1]
7-element Array{Int64,1}:
 1
 3
 3
 3
 2
 1
 1

 julia> mk_class_ids(labels)[2]
3-element Array{String,1}:
 "blue"
 "green"
 "red"
```
"""
function mk_class_ids(labels)

    l = sort(unique(labels))
    ids = [findfirst(x->x==c, l) for c in labels]
    return ids, l
end



"""
    function dataframe_split(df::DataFrames.DataFrame;
                             teaching="y", split=0.8, balanced=true)

Split data, organised row-wise in a DataFrame into train and validation sets.

### Arguments:
+ `df`: data
+ `teaching="y"`: name or index of column with teaching input "y"
+ `split=0.8`: fraction of data to be used for the first returned 
              subdataframe
+ `shuffle=true`: shuffle the rows of the dataframe.
+ `balanced=true`: if `true`, result datasets will be balanced by oversampling.
              Returned datasets will be bigger as expected
              but include the same numbers of samples for each class.
"""
function dataframe_split(df::DataFrames.DataFrame; teaching="y",
                         split=0.8, shuffle=true, balanced=true)

    if shuffle
        df .= df[Random.randperm(DataFrames.nrow(df)),:]
    end
    ((trn,ytrn), (vld,yvld)) = do_split(df, df[:,teaching], at=split)

    if balanced
        (trn,ytrn) = do_balance(trn, ytrn)
        (vld,yvld) = do_balance(vld, yvld)
    end

    return trn, vld
end




