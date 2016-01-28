#                   NESTED SAMPLING MAIN PROGRAM
# (GNU General Public License software, (C) Sivia and Skilling 2006)
# This file was translated to Julia by Eric Davies in 2016.

module MiniNest

export nested_sampling, SamplingObject

# "typically faster and more accurate"
import Base.Math.JuliaLibm: log, log1p

include("./utils.jl")

# abstract interface for sampling objects
abstract SamplingObject

"""
`prior!(obj::SamplingObject)`: set Object according to prior

This is an abstract method. All T<:SamplingObject must implement it.

Args:
- `obj`: SamplingObject being set
"""
function prior!(obj::SamplingObject)
    error("`prior!` not implemented for $(typeof(obj))")
end

"""
`explore!(obj::SamplingObject, logLstar::Float64)`:
Evolve object withing likelihood constraint

This is an abstract method. All T<:SamplingObject must implement it.

Args:
- `obj`: SamplingObject being evolved
- `logLstar`: likelihood constraint L > Lstar
"""
function explore!(obj::SamplingObject, logLstar::Float64)
    error("`explore!` not implemented for $(typeof(obj))")
end

"""
`results(samples::Vector{Object}, logZ::Float64)`: posterior probabilities and summary

This is an abstract method. All T<:SamplingObject must implement it.

Args:
- `samples`: SamplingObjects defining posterior
- `logZ`: Evidence (= total weight = SUM[Samples] Weight)
"""
function results{T<:SamplingObject}(samples::Vector{T}, logZ::Float64)
    error("`results` not implemented for $T")
end


function nested_sampling{T<:SamplingObject}(
    ::Type{T},
    n::Int,
    MAX::Int,
)
    H = 0.0  # Information, initially 0
    logZ = DBL_MIN  # ln(Evidence Z, initially 0)
    objs = [T() for i = 1:n]
    samples = [T() for i = 1:MAX]

    foreach(prior!, objs)

    # outermost interval of prior mass
    logwidth = log(1.0 - exp(-1.0 / n))

    # nested sampling loop
    for i = 1:MAX
        # worst object in collection, with Weight = width * likelihood
        worst = 1
        for j = 2:n
            if objs[j].logL < objs[worst].logL
                worst = j
            end
        end
        objs[worst].logWt = logwidth + objs[worst].logL

        # update Evidence Z and Information H
        logZnew = logadd(logZ, objs[worst].logWt)
        H = exp(objs[worst].logWt - logZnew) * objs[worst].logL +
            exp(logZ - logZnew) * (H + logZ) - logZnew
        logZ = logZnew

        # posterior samples (optional)
        samples[i] = deepcopy(objs[worst])

        # kill worst object in favour of copy of different survivor
        logLstar = objs[worst].logL
        if n > 1
            copy_obj = worst
            while copy_obj == worst
                copy_obj = rand(1:n)
            end
            objs[worst] = objs[copy_obj]
        end

        # evolve copied object within constraint
        explore!(objs[worst], logLstar)

        # shrink interval
        logwidth -= 1.0 / n
    end

    # Exit with evidence Z, information H, and optional posterior samples
    @printf("# iterates = %d\n", MAX)
    @printf("Evidence: ln(Z) = %g ± %g\n", logZ, sqrt(H / n))
    @printf("Information: H = %g nats = %g bits\n", H, H / log(2.0))
    results(samples, logZ)
end

end # module
