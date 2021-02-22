# Attention mechanisms:
#
"""
    abstract type AttentionMechanism

Attention mechanisms follow the same interface and common signatures.

If possible, the algorithm allows precomputing of the projections
of the context vector
generated by the encoder in a encoder-decoder-architecture
(i.e. in case of an RNN encoder the accumulated encoder hidden states).

By default attention scores are scaled according to
Vaswani et al., 2017 *(Vaswani et al., Attention Is All You Need,
CoRR, 2017)*.

All algorithms use soft attention.

## Constructors:
    Attn*Mechanism*(dec_units, enc_units; scale=true)
    Attn*Mechanism*(units; scale=true)

The one-argument version can be used, if encoder dimensions and decoder
dimensions are the same.

## Signatures:
    function (attn::AttentionMechnism)(h_t, h_enc; reset=false)
    function (attn::AttentionMechanism)(; reset=false)

### Arguments:
+ `h_t`:    decoder hidden state. If `h_t` is a vector, its length
            equals the number of decoder units. If it is a matrix,
            `h_t` includes the states for a minibatch of samples and has
            tha size [units, mb].
+ `h_enc`:  encoder hidden states, 2d or 3d. If `h_enc` is a
            matrix [units, steps] with the hidden states of all encoder steps.
            If 3d [units, mb, steps] encoder states for a minibatch is
            included.
+ `reset=false`: If the keyword argument is set to `true`, projections of
            the encoder states are computed. By default projections are
            stored in the object and reused until the object is resetted.
            For attention mechanisms that don't allow precomputation
            the argument is ignored.

The short form `(::AttentionMechanism)(reset=true)` can be used to reset
the precomputed projections.

## Attention Mechanisms:

All attention mechanisms calculate attention factors α from scores
derived from projections of the encoder hidden states:
```math
\\alpha = \\mathrm{softmax}(\\mathrm{score}(h_{enc},h_{t}) \\cdot 1/\sqrt{n}))
```

or scaled:

```math
\\alpha = \\mathrm{softmax}(\\mathrm{score}(h_{enc},h_{t}) \\cdot \\nicefrac{1}{\\sqrt{n}})
```

The following attention mechanisms are implemented:
"""
abstract type AttentionMechanism
end

function (attn::AttentionMechanism)(;reset=false)

    if reset && hasfield(typeof(attn), :projections)
        attn.projections = nothing
    end
end


"""
    mutable struct AttnBahdanau <: AttentionMechanism

Bahdanau-style (additive, concat) attention mechanism according
to the paper:

*D. Bahdanau, KH. Co, Y. Bengio,
Neural Machine Translation by jointlylearning to align and translate,
ICLR, 2015*.

```math
\\mathrm{score}(h_{t},h_{enc}) = v_{a}^{\\top}\\cdot\\tanh(W[h_{t},h_{enc}])
```

### Constructors:
    AttnBahdanau(dec_units, enc_units; scale=true)
    AttnBahdanau(units; scale=true)
"""
mutable struct AttnBahdanau <: AttentionMechanism
    enc
    dec
    combine
    scale
    projections
    AttnBahdanau(dec_units, enc_units; scale=true) = new(Linear(enc_units, dec_units, bias=false),
                                Linear(dec_units, dec_units, bias=false),
                                Linear(dec_units, 1, bias=false),
                                scale ? 1/sqrt(enc_units) : 1.0,
                                nothing)
    AttnBahdanau(units; scale=true) = AttnBahdanau(units, units,
                                scale=scale)
end

function (attn::AttnBahdanau)(h_t, h_enc; reset=false)
                                    # h_t is a (n_units, <n_mb>),
                                    # h_enc is (n_units, <n_mb>, n_steps)
    # make all 3d:
    #
    h_encR = reshape(h_enc, size(h_enc)[1], :, size(h_enc)[ndims(h_enc)])
    units, mb, steps = size(h_encR)
    h_tR = reshape(h_t, size(h_t)[1], :)

    # this is possible, because the Linear-layers are used:
    #
    if reset || attn.projections == nothing
        attn.projections = attn.enc(h_encR)
    end
    score = attn.combine(tanh.(attn.projections .+ attn.dec(h_tR)))
    score *= attn.scale
    α = softmax(score, dims=3)

    # calc. context from encoder states:
    #
    c = sum(α .* h_encR, dims=3)

    # remove unneeded dims:
    #
    c = reshape(c, units, mb)
    α = reshape(α, mb, steps)
    return c, α
end

"""
    mutable struct AttnLuong <: AttentionMechanism

Luong-style (multiplicative) attention mechanism according to
the paper (referred as *General*-type attention):
*M.-T. Luong, H. Pham, C.D. Manning,
Effective Approaches to Attention-based Neural Machine Translation,
CoRR, 2015*.
```math
\\mathrm{score}(h_{t},h_{enc}) = h_{t}^{\\top} W h_{enc}
```

### Constructors:
    AttnLuong(dec_units, enc_units; scale=true)
    AttnLuong(units; scale=true)
"""
mutable struct AttnLuong <: AttentionMechanism
    enc
    scale
    projections
    AttnLuong(dec_units, enc_units; scale=true) = new(Linear(enc_units,
                dec_units, bias=false),
                scale ? 1/sqrt(enc_units) : 1.0,
                nothing)
    AttnLuong(units; scale=true) = AttnLuong(units, units, scale=scale)
end

function (attn::AttnLuong)(h_t, h_enc; reset=false)
    # make all 3d:
    #
    h_encR = reshape(h_enc, size(h_enc)[1], :, size(h_enc)[ndims(h_enc)])
    units, mb, steps = size(h_encR)
    h_tR = reshape(h_t, size(h_t)[1], :)

    if reset || attn.projections == nothing
        attn.projections = attn.enc(h_encR)
    end

    score = sum(attn.projections .* h_tR, dims=1)
    score *= attn.scale
    α = softmax(score, dims=3)

    # calc. context from encoder states:
    #
    c = sum(α .* h_encR, dims=3)

    # remove unneeded dims:
    #
    c = reshape(c, units, mb)
    α = reshape(α, mb, steps)
    return c, α
end



"""
    mutable struct AttnDot <: AttentionMechanism

Dot-product attention (without trainable parameters)
according to the Luong, et al. (2015) paper.

``\\mathrm{score}(h_{t},h_{enc}) = h_{t}^{\\top} h_{enc}``

### Constructors:
    AttnDot(; scale=true)
"""
mutable struct AttnDot <: AttentionMechanism
    scale
    AttnDot(;scale=true) = new(scale ? 1/sqrt(enc_units) : 1.0)
end

function (attn::AttnDot)(h_t, h_enc; reset=false)
    # make all 3d:
    #
    h_encR = reshape(h_enc, size(h_enc)[1], :, size(h_enc)[ndims(h_enc)])
    units, mb, steps = size(h_encR)
    h_tR = reshape(h_t, size(h_t)[1], :)

    score = sum(h_encR .* h_tR, dims=1)
    score *= attn.scale
    α = softmax(score, dims=3)

    # calc. context from encoder states:
    #
    c = sum(α .* h_encR, dims=3)

    # remove unneeded dims:
    #
    c = reshape(c, units, mb)
    α = reshape(α, mb, steps)
    return c, α
end

#
# + `AttnLocation`: Location-based attention
#         according to the Luong, et al. (2015) paper.
#         ```math
#         \\mathrm{score}(h_{t}) = W h_{t}
#         ```
#
