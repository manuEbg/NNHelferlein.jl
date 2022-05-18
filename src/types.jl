"""
    abstract type Layer end

Mother type for layers hierarchy.
"""
abstract type Layer
end


""" 
    abstract type RecurrentUnit end

Supertype for all recurrent unit types.
Self-defined recurrent units which are a child of `RecurrentUnit`
can be used inside the 'Recurrent' layer.

### Interface
All subtypes of `RecurrentUnit` must provide the followning:
+ a constructor with signature `Type(n_inputs, n_units; kwargs)` and
    arbitrary keyword arguments.
+ an implementation of signature `(o::Recurrent)(x)`
    where `x` is a 3d- or 2d-array of shape [fan-in, mb-size, 1] or 
    [fan-in, mb-size].
    The function must return the result of one forward 
    computation for one step and return the hidden state
    and set the internal fields `h` and optionally `c`.
+ a field `h` (to store the last hidden state)
+ an optional field `c`, if the cell state is to be stored
    such as in a lstm unit.
"""
abstract type RecurrentUnit end




"""
    abstract type DataLoader

Mother type for minibatch iterators.
"""
abstract type DataLoader end

"""
    struct SequenceData <: DataLoader

Type for a generic minibatch iterator.
All NNHelferlein models accept minibatches of type `DataLoader`.

### Constructors:

    SequenceData(x; shuffle=true)

+ `x`: List, Array or other iterable object with the minibatches
+ `shuffle`: if `true`, minibatches are shuffled every epoch.
"""
mutable struct SequenceData <: DataLoader
    mbs
    l
    indices
    shuffle
    SequenceData(x; shuffle=true) = new(x, length(x), collect(1:length(x)), shuffle)
end

function Base.iterate(it::SequenceData, state=0)

    # shuffle if first call:
    #
    if it.shuffle && state == 0
        it.indices = Random.randperm(it.l)
    end
        
    if state >= it.l
        return nothing
    else
        state += 1
        return it.mbs[it.indices[state]], state
    end
end

Base.length(it::SequenceData) = it.l
Base.eltype(it::SequenceData) = eltype(first(it.mbs))


"""
    struct PartialIterator <: DataLoader

The `PartialIterator` wraps any iterator and will only iterate the states
specified in the list `indices`. 

### Constuctors

    PartialIterator(inner, indices; shuffle=true) 

Type of the states must match
the states of the wrapped iterator `inner`. A `nothing` element may be 
given to specify the first iterator element.

If `shuffle==true`, the list of indices are shuffled every time the
`PartialIterator` is started.
"""
mutable struct PartialIterator <: DataLoader
    inner
    indices
    l
    shuffle
    PartialIterator(inner, indices; shuffle=true) = new(inner, indices, length(indices), shuffle)
end

function Base.iterate(it::PartialIterator, state=0)
    
    if it.shuffle && state == 0
        Random.shuffle!(it.indices)
    end
    
    if state >= it.l
        return nothing
    else
        state += 1
        inner_state = it.indices[state]
        
        if isnothing(inner_state)
            return iterate(it.inner,)[1], state
        else
            return iterate(it.inner, inner_state)[1], state
        end
    end
end

Base.length(it::PartialIterator) = it.l
Base.eltype(it::PartialIterator) = eltype(first(it.inner))


""" 
    type MBNoiser

Iterator to wrap any Knet.Data iterator of minibatches in 
order to add random noise.    
Each value will be multiplied with a random value form 
Gaussian noise with mean=1.0 and sd=sigma.

### Construtors:
    MBNoiser(mbs::Knet.Data, σ)
    MBNoiser(mbs::Knet.Data; σ=1.0)

+ `mbs`: iterator with minibatches
+ `σ`: standard deviation for the Gaussian noise

### Example:
```juliaREPL
julia> trn = minibatch(x)
julia> tb_train!(mdl, Adam, MBNoiser(trn, σ=0.1))
julia> mbs_noised = MBNoiser(mbs, 0.05)
```
"""
struct MBNoiser  <: DataLoader
    mbs::Union{Knet.Data, DataLoader}
    size
    σ
    MBNoiser(mbs::Union{Knet.Data, DataLoader}, sd=1.0; σ=sd) = new(mbs, size(first(mbs)[1]), σ)
end



# first call:
#
function Base.iterate(nr::MBNoiser) 
    return iterate(nr,0)
end

# subsequent calls with state:
#
function Base.iterate(nr::MBNoiser, state)
    next_inner = iterate(nr.mbs, state)
    if isnothing(next_inner)
        return nothing
    else
        next_mb, next_state = next_inner
        return (next_mb[1] .* convert2KnetArray(randn(nr.size) .* nr.σ .+ 1) , next_mb[2]), 
                next_state
    end
end

# and length = length of inner iterator:
#
Base.length(it::MBNoiser) = length(it.mbs)





"""
    struct MBMasquerade  <: DataLoader

Iterator wrapper to partially mask training data of a minibatch 
iterator of type `Knet.Data` or `NNHelferlein.DataLoader`.

### Constructors:
    Masquerade(it, rho=0.1; mode=:noise, value=0.0)
    Masquerade(it; ρ=0.1, mode=:noise, value=0.0)

The constructor may be called with the density `rho` as normal
argument or `ρ` as keyword argument.

### Arguments:
+ `it`: Minibatch iterator the must deliver (x,y)-tuples of 
        minibatches
+ `ρ=0.1` or `rho`: Density of mask; a value of 1.0 will mask everything,
        a value of 0.0 nothing.
+ `value=0.0`: the value with which the masking is done.
+ `mode=:noise`: type of masking:
        * `:noise`: randomly distributed single values of the 
            training data will be overwitten with `value`.
        * `:patch`: a single rectangular region will be 
            overwritten.

### Examples:

```juliaREPL
julia> dtrn 
26-element Knet.Train20.Data{Tuple{CuArray{Float32}, Array{UInt8}}}

julia> mtrn = Masquerade(dtrn, 0.5, value=2.0, mode=:patch)
Masquerade(26-element Knet.Train20.Data{Tuple{CuArray{Float32}, Array{UInt8}}}, 0.5, 2.0, :patch)

julia> dmask = Masquerade(dtrn, ρ=0.3) |> x->Masquerade(x, ρ=0.3, value=1.0)
Masquerade(Masquerade(26-element Knet.Train20.Data{Tuple{CuArray{Float32}, Array{UInt8}}}, 0.3, 0.0, :noise), 0.3, 1.0, :noise)
```
"""
struct MBMasquerade  <: DataLoader
    it::Union{Knet.Data, DataLoader}
    ρ
    value
    mode
    Masquerade(it::Union{Knet.Data, DataLoader}, rho=0.1; ρ=rho, mode=:noise, value=0.0) = 
        new(it, ρ, value, mode)
end

function Base.iterate(it::MBMasquerade)
    return iterate(it, 0)
end

function Base.iterate(it::MBMasquerade, state)
    
    next_inner = iterate(it.it, state)
    if isnothing(next_inner)
        return nothing
    end
    
    (x,y), next_state = next_inner
    
    if it.mode == :noise
        x = do_mask(x, it.ρ, it.value)
    elseif it.mode == :patch
        x = do_patch(x, it.ρ, it.value)
    end
    
    return (ifgpu(x),y), next_state
end

Base.length(it::MBMasquerade) = length(it.it)



function do_mask(x, ρ, value)
    
    mask = rand(size(x)...) .< ρ
    x[mask] .= value
    return x
end

function do_patch(x, ρ, value)
    
    p_size = size(x) .* ρ .|> ceil .|> Int
    p_start = [rand(collect(1:s-p+1)) for (s,p) in zip(size(x), p_size) ]
    p_end = p_start .+ p_size .- 1
    ranges = ((i:j) for (i,j) in zip(p_start, p_end))

    x[ranges...] .= value
    return x
end