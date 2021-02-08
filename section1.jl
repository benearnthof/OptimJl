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

