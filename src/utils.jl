# smallest Float64 greater than -Inf
const DBL_MIN = nextfloat(-Inf)

"""
`uniform()`: random Float64 in the interval (0,1)
"""
uniform() = (rand(UInt) + 0.5) / (typemax(UInt) + 1.0)

"""
`logadd(x, y)`: logarithmic addition log(exp(x)+exp(y))
"""
logadd(x, y) = x > y ? (x + log1p(exp(y - x))) : (y + log1p(exp(x - y)))
