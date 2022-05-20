# import Pkg; Pkg.add("Documenter")#; Pkg.add("NNHelferlein")
using Documenter, NNHelferlein

makedocs(modules = [NNHelferlein],
         clean = false,
         sitename = "NNHelferlein.jl",
         authors = "Andreas Dominik",
         pages = [
                  "Introduction" => "index.md",
                  "Overview" => "overview.md",
                  "Examples" => "examples.md",
                  "API Reference" => "api.md",
                  "License" => "license.md"
                  ],
         )

deploydocs(
    repo   = "github.com/KnetML/NNHelferlein.jl.git",
    target = "build",
    branch = "gh-pages",
    # deps   = nothing | <Function>,
    # make   = nothing | <Function>,
    devbranch = "main",
    # devurl = "dev",
    versions = ["stable" => "v^", "v#.#", devurl => "dev"],
    # push_preview    = false,
    # repo_previews   = repo,
    # branch_previews = branch
)
