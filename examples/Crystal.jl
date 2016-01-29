#            "PSEUDO-CRYSTAL" NESTED SAMPLING APPLICATION
# (GNU General Public License software, (C) Sivia and Skilling 2006)
# translated to Julia by Eric Davies in 2016
# Problem:     M switches s = 0 or 1, grouped in clusters of widths h.
#                 e.g. M=10, s = {0,0,0,1,1,1,1,0,0,1}
#                            h = {  3  ,   4   , 2 ,1}
# Inputs:
#  Prior(s)    is uniform, 1/2^N on each of 2^M states
#  Likelihood  is L(s) = exp( SUM h(h-1)/M )
# Outputs:
#  Evidence    is Z = SUM L(s) Prior(s)
#  Posterior   is P(s) = L(s) Prior(s)/ Z
#  Information is H = SUM P(s) log(P(s)/Prior(s))

module Crystal

using MiniNest

import MiniNest: uniform

const NUM_SWITCHES = 1000

type CrystalObject <: SamplingObject
    "state of switches"
    states::Array{Bool}

    "logLikelihood = ln prob(data | s)"
    logL::Float64

    "log(Weight), adding to SUM(Wt) = Evidence Z"
    logWt::Float64

    CrystalObject() = new(Array{Bool}(NUM_SWITCHES), 0.0, 0.0)
end

function log_likelihood(state::Array{Bool})
    logL = 0.0
    i = 1  # LH boundary
    for j = 2:NUM_SWITCHES
        if state[j] != state[j - 1]  # RH boundary found
            logL += (j - i) * (j - i - 1)  # cluster width h = j - i
            i = j  # reset LH boundary
        end
    end
    logL += (NUM_SWITCHES + 1 - i) * (NUM_SWITCHES + 1 - i - 1)  # rightmost cluster

    # normalized jitter eliminates ties between likelihood values
    return logL / NUM_SWITCHES + sqrt(eps(Float64)) * uniform()
end

function MiniNest.prior!(obj::CrystalObject)
    rand!(obj.states)
    obj.logL = log_likelihood(obj.states)
end

function MiniNest.explore!(obj::CrystalObject, logLstar::Float64)
    for i = 1:(10 * NUM_SWITCHES)
        trial = rand(1:NUM_SWITCHES)
        obj.states[trial] = !obj.states[trial]  # try flipping
        logLtrial = log_likelihood(obj.states)
        if logLtrial > logLstar
            obj.logL = logLtrial  # accept
        else
            obj.states[trial] = !obj.states[trial]
        end
    end
end

function MiniNest.results(samples::Vector{CrystalObject}, logZ::Float64)
    for k = 1:length(samples)
        @printf("%7.2f %8.4f\n", -k, samples[k].logL)
    end
end

function main()
    nested_sampling(CrystalObject, 1, 800)
end

end

Crystal.main()
@time Crystal.main()
