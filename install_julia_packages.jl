# Rather than release the whole repo as a julia library, I've just
# made a list of the required packages here.

# This is only designed for Julia 0.3
@assert VERSION < v"0.4.0-dev"

required_packages =
	["DataFrames", "Distributions", "JSON",
	 "JuMP", "Ipopt", "ReverseDiffSparse", "Compat"]

# Pkg.available() doesn't pick up cloned packages.
available_packages = keys(Pkg.installed())

for package in setdiff(required_packages, available_packages)
	Pkg.add(package)
end

# This package hasn't been released yet.
if !("DataFramesIO" in available_packages)
	Pkg.clone("https://github.com/johnmyleswhite/DataFramesIO.jl")
end
