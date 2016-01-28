#            "LIGHTHOUSE" NESTED SAMPLING APPLICATION
# (GNU General Public License software, (C) Sivia and Skilling 2006)
# translated to Julia by Eric Davies in 2016
#              u=0                                 u=1
#               -------------------------------------
#          y=2 |:::::::::::::::::::::::::::::::::::::| v=1
#              |::::::::::::::::::::::LIGHT::::::::::|
#         north|::::::::::::::::::::::HOUSE::::::::::|
#              |:::::::::::::::::::::::::::::::::::::|
#              |:::::::::::::::::::::::::::::::::::::|
#          y=0 |:::::::::::::::::::::::::::::::::::::| v=0
# --*--------------*----*--------*-**--**--*-*-------------*--------
#             x=-2          coastline -->east      x=2
# Problem:
#  Lighthouse at (x,y) emitted n flashes observed at D[.] on coast.
# Inputs:
#  Prior(u)    is uniform (=1) over (0,1), mapped to x = 4*u - 2; and
#  Prior(v)    is uniform (=1) over (0,1), mapped to y = 2*v; so that
#  Position    is 2-dimensional -2 < x < 2, 0 < y < 2 with flat prior
#  Likelihood  is L(x,y) = PRODUCT[k] (y/pi) / ((D[k] - x)^2 + y^2)
# Outputs:
#  Evidence    is Z = INTEGRAL L(x,y) Prior(x,y) dxdy
#  Posterior   is P(x,y) = L(x,y) / Z estimating lighthouse position
#  Information is H = INTEGRAL P(x,y) log(P(x,y)/Prior(x,y)) dxdy

module Lighthouse

using MiniNest

import MiniNest: uniform

type LighthouseObject <: SamplingObject
    "Uniform-prior controlling parameter for x"
    u::Float64

    "Uniform-prior controlling parameter for y"
    v::Float64

    "Geographical easterly position of lighthouse"
    x::Float64

    "Geographical northerly position of lighthouse"
    y::Float64

    "logLikelihood = ln Prob(data | position)"
    logL::Float64

    "log(Weight), adding to SUM(Wt) = Evidence Z"
    logWt::Float64

    LighthouseObject() = new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function set_to!(obj1::LighthouseObject, obj2::LighthouseObject)
    obj1.u = obj2.u
    obj1.v = obj2.v
    obj1.x = obj2.x
    obj1.y = obj2.y
    obj1.logL = obj2.logL
    obj1.logWt = obj2.logWt
end

map_to_x(u) = 4.0 * u - 2.0
map_to_y(v) = 2.0 * v

const D = [ 4.73,  0.45, -1.73,  1.09,  2.19,  0.12,
          1.31,  1.00,  1.32,  1.07,  0.86, -0.49, -2.59,  1.73,  2.11,
          1.61,  4.98,  1.71,  2.23,-57.20,  0.96,  1.25, -1.56,  2.45,
          1.19,  2.17,-10.66,  1.91, -4.16,  1.92,  0.10,  1.98, -2.51,
          5.55, -0.47,  1.91,  0.95, -0.78, -0.84,  1.72, -0.01,  1.48,
          2.70,  1.21,  4.41, -4.79,  1.33,  0.81,  0.20,  1.58,  1.29,
         16.19,  2.75, -2.38, -1.79,  6.50,-18.53,  0.72,  0.94,  3.64,
          1.94, -0.11,  1.57,  0.57 ]
const N = length(D)

"""
`log_likelihood(x::Float64, y::Float64)`: log likelihood of a position

Args:
- `x`: Easterly position
- `y`: Northerly position
"""
function log_likelihood(x::Float64, y::Float64)
    logL = 0.0
    for k = 1:N
        logL += log((y / 3.1416) / ((D[k] - x) ^ 2 + y ^ 2))
    end

    return logL
end

function MiniNest.prior!(obj::LighthouseObject)
    obj.u = uniform()
    obj.v = uniform()
    obj.x = map_to_x(obj.u)
    obj.y = map_to_y(obj.v)
    obj.logL = log_likelihood(obj.x, obj.y)
end

function MiniNest.explore!(obj::LighthouseObject, logLstar::Float64)
    step = 0.1  # initial guess of suitable step size in (0,1)
    m = 20  # MCMC number of steps
    accept = 0  # number of MCMC acceptances
    reject = 0  # number of MCMC rejections
    trial = LighthouseObject()  # trial object

    for i = 1:m
        trial.u = obj.u + step * (2.0 * uniform() - 1.0)  # |move| < step
        trial.v = obj.v + step * (2.0 * uniform() - 1.0)  # |move| < step

        trial.u -= floor(trial.u)  # wraparound to stay within (0,1)
        trial.v -= floor(trial.v)  # wraparound to stay within (0,1)

        trial.x = map_to_x(trial.u)
        trial.y = map_to_y(trial.v)

        trial.logL = log_likelihood(trial.x, trial.y)

        # accept iff within hard likelihood constraint
        if trial.logL > logLstar
            set_to!(obj, trial)
            accept += 1
        else
            reject += 1
        end

        if accept > reject
            step *= exp(1.0 / accept)
        elseif accept < reject
            step /= exp(1.0 / reject)
        end
    end
end

function MiniNest.results(samples::Vector{LighthouseObject}, logZ::Float64)
    x = xx = 0.0  # 1st and 2nd moments of x
    y = yy = 0.0  # 1st and 2nd moments of y
    w = 0.0  # proportional weight

    for i = 1:length(samples)
        w = exp(samples[i].logWt - logZ)
        x += w * samples[i].x
        xx += w * samples[i].x ^ 2
        y += w * samples[i].y
        yy += w * samples[i].y ^ 2
    end

    @printf("mean(x) = %g, stddev(x) = %g\n", x, sqrt(xx - x^2))
    @printf("mean(y) = %g, stddev(y) = %g\n", y, sqrt(yy - y^2))
end

function main()
    nested_sampling(LighthouseObject, 100, 1000)
end

end

Lighthouse.main()
