# Optimization Algorithms in Julia
# Algorithms for Optimization by Mykel J. Kochenderfer, Tim A. Wheeler
# Derivatives and Gradients

# Limit equation defining the derivative can be presented in three different ways: 
# forward difference, central difference, and backward difference
# if f can be represented symbolically, we can use symbolic differentiation 
using SymEngine
@vars x; # we define x as a symbolic variable
f = x^2 + x/2 - sin(x)/x
diff(f, x)

# symbolic variables are defined with _symbol 
@vars y, x 
g = x^2 + y^3 - x*y^2
diff(g, x)
diff(g, y)
diff(g, x, y)
# differentiates wrt x first then wrt y
# Gradient points in direction of steepest ascent of the tangent hyperplane
# gradient of f at x is written ∇f(x) and is a vector. 
# each component of that vector is the partial derivative of f wrt that component 
# ∇f(x) = [δf(x)/δx1, ..., δf(x)/δxn]

# The hessian of a multivariate function is a matrix containing all the second Derivatives
# wrt the input. => captures informatio nabout the local curvature of the function
# ∇²f(x) = [δ²f(x)/δx1δx1, ..., δ²f(x)/δxnδxn]

# computing gradient of f at [2,0]
f = x * sin(y) + 1
∇f = [diff(f, x), diff(f, y)]

f(x => 2, y => 0)
∇f
grad_f = [∇f[i](x => 2, y => 0) for i = 1:size(∇f)[1]]

# directional Derivatives
# the directional derivative of a function f at point x in direction s is 
# ∇f(x)' * s
# the dot product of the gradient evaluated at x and the direction vector s
@vars x, y
f = x * y
function symbolicgrad(fun)
    [diff(fun, x), diff(fun, y)]
end 

∇f = symbolicgrad(f)

function evalgrad(fun, x1, x2)
    [fun[i](x => x1, y => x2) for i = 1:size(∇f)[1]]
end

grad_f = evalgrad(∇f, 1, 0)
# to compute the directional derivative we take the dot product
s = [-1, -1]

function dot(x, y) 
    sum(x .* y)
end

dir_f = dot(grad_f, s)

function dir_deriv(fun, at, dir)
    ∇f = symbolicgrad(fun)
    grad_f = evalgrad(∇f, at[1], at[2])
    dot(grad_f, dir)
end

dir_deriv(f, [1, 0], [-1, -1])

# numerical differenciation
# estimating derivatives numerically => derive from function evaluation 
# evaluate the function at multiple points near the point of interest and take averages
# function diff_forward(fun, x1; h = sqrt(eps(Float64)))
#     (fun(x => (x1+h)) - fun(x => x1))/h
# end

diff_forward(f, x; h = sqrt(eps(Float64))) = (f(x + h) - f(x))/h
diff_central(f, x; h = cbrt(eps(Float64))) = (f(x+h/2) - f(x - h/2))/h
diff_backward(f, x; h = sqrt(eps(Float64))) = (f(x) - f(x - h))/h

h(x) = (x^2)

diff_forward(h, 1)
diff_backward(h, 1)
diff_central(h, 1)

# deriving these functions from the taylor expansion of f(x) at f(x + h), we can show that
# the error term of the forward method is of the order O(h) => linear error
# the central difference method has error of order O(h²)

# 2.4 Automatic differentiation 
# key to automatic differentiation techniques is the application of the chain rule
# if we apply the chain rule to nested function recursively we can automate differentiation 
# using a computational graph
# nodes are operations and edges are input output relations
# two methods: Forward accumulation and backward accumulation 
# we pass in a dual number into a function
# it returns the evaluation of the function and its derivative

# reverse accumulation also yields these results but only requires a forward pass to 
# calculate intermediate values and only a single backward pass to calculate the 
# corresponding multidimensional gradient. 

# an example of dual numbers 
struct Dual
    v
    ∂
end

# implementing methods for the Dual struct
Base.:+(a::Dual, b::Dual) = Dual(a.v + b.v, a.∂ + b.∂)
Base.:*(a::Dual, b::Dual) = Dual(a.v*b.v, a.v*b.∂ + b.v*a.∂)
Base.log(a::Dual) = Dual(log(a.v), a.∂/a.v)

function Base.max(a::Dual, b::Dual)
    v = max(a.v, b.v)
    ∂ = a.v > b.v ? a.∂ : a.v < b.v ? b.∂ : NaN
    return Dual(v,∂)
end

function Base.max(a::Dual, b::Int) 
    v = max(a.v, b) 
    ∂ = a.v > b ? a.∂ : a.v < b ? 0 : NaN
    return Dual(v,∂)
end

a = Dual(3,1)
b = Dual(2,0)
log(a * b + max(a, 2))

# all these methods and many more are already implemented in the package ForwardDiff
using ForwardDiff
a = ForwardDiff.Dual(3,1)
b = ForwardDiff.Dual(2,0)
log(a * b + max(a, 2))

# backward accumulation is provided in the Zygote package 
import Zygote: gradient
f_1(a, b) = log(a * b + max(a, 2))
gradient(f_1, 3, 2)

# final note: Very small steps h can result in subtractive cancellation
# the complex step method may help, but other methods to ameliorate this can also be applied

# exercises: 2.3
f_2(x) = log(x) + exp(x) + 1/x
gradient(f_2, 0)
@vars x
SymEngine::diff(f_2, x)

@vars x; # we define x as a symbolic variable
f = log(x) + exp(x) + 1/x
diff(f, x)
∂x = diff(f, x)
∂x(x => 0)
# the log term dominates the gradient because evaluating x^(-1) at 0 results in Inf 
