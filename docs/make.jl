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
                  "License" => "license.md",
                  hide("Changelog" => "changelog.md")
                  ],
         )

deploydocs(
    repo   = "github.com/KnetML/NNHelferlein.jl.git")
#    target = "build",
#    branch = "gh-pages",
#    devbranch = "main",
#    # devurl = "dev",
#    versions = ["stable" => "v^", "v#.#", "dev" => "dev"]
#)
