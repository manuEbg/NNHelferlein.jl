using Pkg
Pkg.add("DataFrames")
using DataFrames

# dataframes:
#
function test_read_df()
    df = dataframe_read(joinpath("data", "iris150.csv"))
    return nrow(df) == 150
end

function test_df_loader()

        trn = DataFrame(x1=randn(16), x2=randn(16),
                        x3=randn(16), x4=randn(16),
                        x5=randn(16), x6=randn(16),
                        x7=randn(16), x8=randn(16),
                        y=["blue", "red", "green", "green",
                           "blue", "red", "green", "green",
                           "blue", "red", "green", "green",
                           "blue", "red", "green", "green"])

        mb1 = dataframe_minibatches(trn, size=4, teaching="y", ignore="x1")

        trn.iy = [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4]
        @show mb2 = dataframe_minibatches(trn, size=4, teaching="iy", ignore=["x1", "y"])
        @show first(mb2)[2]

        trn.is = [:a,:a,:a,:a,:b,:b,:b,:b,:c,:c,:c,:c,:d,:d,:d,:d]
        mb3 = dataframe_minibatches(trn, size=4, teaching="is", ignore=["iy", "y"])

        return first(mb1)[2] == UInt8[0x01  0x03  0x02  0x02] &&
               first(mb2)[2] == UInt8[0x01 0x01 0x01 0x01]    &&
               isnothing(mb3)
end



function test_df_split()
    df = dataframe_read(joinpath("data", "iris150.csv"))
    t,v = dataframe_split(df, split=0.5, shuffle=true,
          teaching="species", balanced=true)
    return nrow(v) == 75
end

function test_df_class_ids()
    df = dataframe_read(joinpath("data", "iris150.csv"))
    c = mk_class_ids(df.species)
    return c[2] == ["setosa", "versicolor", "virginica"]
end

function test_df_minibatch()
    df = dataframe_read(joinpath("data", "iris150.csv"))
    mb = dataframe_minibatches(df, size=10, teaching="species")
    return size(first(mb)[1]) == (4,10)
end

function test_df_errors()
    df1 = dataframe_read("../data/iris/iris150.ods")
    df2 = dataframe_read("../data/iris/iris150.dat")

    return isnothing(df1) &&
           isnothing(df2)
end

# test NLP utils:
#
function test_tokenizer()
    tok = WordTokenizer(["I love Julia",
                         "Peter loves Python",
                         "We all marvel Geoff"])
    l = tok(["I love Julia", "Peter loves Python", "We all marvel Geoff"],
            add_ctls=true)

    sentence = tok([8, 12, 5])
    return tok("Julia") == 8 && tok(8) == "Julia" &&
           l[3] == [1, 6, 14, 13, 11, 2] &&
           sentence == "Julia loves Peter"
end

function test_seq_mb()
    tok = WordTokenizer(["I love Julia",
                         "Peter loves Python",
                         "We all marvel Geoff"])

    t = [tok("I love Julia", split_words=true),
         tok("Peter loves Python", split_words=true),
         tok("Peter loves Julia and Scala", split_words=true)]

    mb = sequence_minibatch(t, 2, shuffle=false)
    return size(first(mb)) == (3,2)
end

function test_seq_mb_xy()
    tok = WordTokenizer(["I love Julia",
                         "Peter loves Python",
                         "We all marvel Geoff"])

    t = [tok("I love Julia", split_words=true),
         tok("Peter loves Python", split_words=true),
         tok("Peter loves Julia and Scala", split_words=true)]

    y = [1,1,2]
    mb = sequence_minibatch(t, y, 2, partial=true, shuffle=false)
    o = [m[1] for m in mb]
    return first(mb)[2] == [1;1] && length(o) == 2
end

function test_pad()
    s = [5,3,2]
    s = pad_sequence(s, 10)
    return length(s) == 10
end

function test_trunc()
    s = [9,5,7,8,9,6,7,9]
    s = truncate_sequence(s, 3, end_token=20)
    return length(s) == 3 && s[3] == 20
end



# test tatoeba on tiny dataset Galician:
#
function test_tatoeba()
    en,glg = get_tatoeba_corpus("glg")
    return en isa AbstractArray
end

function test_seq2seq_mb()
     en,glg = get_tatoeba_corpus("glg")
     ven = WordTokenizer(en)
     vglg = WordTokenizer(glg)
     mb = sequence_minibatch(ven.(en, split_words=true),
            vglg.(glg, split_words=true),
            32, shuffle=true)

    mb1 = first(mb)
    return mb1 isa Tuple
end


# test noiser:
#
using Statistics
function test_noiser()
    x = ones(Float32, 64, 1000)
    y = rand(1:10, 1000)
    mbs = minibatch(x, y, 128)

    noise = MBNoiser(mbs)
    d = first(noise)[1]

    i = 0
    for (x,y) in noise
        i += 1
    end
    
    return i == length(noise) 
end

test_noiser()

function test_masquerade()
    x = zeros(Float32, 100,100,20)
    y = rand(1:10, 20)
    mbs = minibatch(x, y, 10)

    # test mode noise:
    #
    masked = MBMasquerade(mbs, ρ=0.5, value=1.0, mode=:noise)
    (x,y) = first(masked)
    @show n = sum(x)

    # test mode patch:
    #
    masked = MBMasquerade(mbs, ρ=0.5, value=1.0, mode=:patch)
    (x,y) = first(masked)
    @show p = sum(x)

    return isapprox(n, 50000, atol=2000) && isapprox(p, 50000, atol=2000)
end



function test_split_mbs()
    x = ones(Float32, 64, 1000)
    y = rand(1:10, 1000)
    mbs = minibatch(x, y, 128)
    
    dtrn, dvld = split_minibatches(mbs, 0.8)
    tl1 = length(dtrn)    # 6
    vl1 = length(dvld)    # 1

    dtrn, dvld = split_minibatches(mbs, 0.0)
    tl2 = length(dtrn)    # 1, not 0 because never 0!
    vl2 = length(dvld)    # 6
    
    return tl1 == 6 && vl1 == 1 && tl2 == 1 && vl2 == 6
end



function test_partial_itr()
    x = ones(Float32, 64, 1000)
    y = rand(1:10, 1000)
    mbs = minibatch(x, y, 128)
    
    dtrn, dvld = split_minibatches(mbs, 0.8)

    n = 0
    for (i,j) in dtrn
        n += 1
    end

    return n == 6
end