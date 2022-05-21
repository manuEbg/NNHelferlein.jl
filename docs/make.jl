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
