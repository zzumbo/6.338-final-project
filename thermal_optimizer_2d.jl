using DifferentialEquations
using LinearAlgebra
using Plots
using ForwardDiff
using ProgressMeter
using DiffEqOperators

N = 4
A = zeros(N,N)

x_size = 10
x = zeros(10,10)

# generate_u_prime(x, dx)

function generate1DStencil!(A, N, Δx)
    # Generate the centered difference matrix manually
    for i in 1:N, j in 1:N

    if abs(i-j)<=1
        (A[i,j]+=1)
    end
    if i==j
        (A[i,j]-=3)
    end

    end

    A = A/(Δx^2)
end

function double_partial(x)
    dx = zeros(size(x))
    for i in 2:length(dx)-1
      dx[i] = -1*x[i-1] + 2*x[i] + -1*x[i+1]
    end
    dx[1] = (2*x[1] + -1*x[2])
    dx[end] = (-1*x[end-1] + 2*x[end])
end

function double_partial_2D(x)


end

dx = 1.0
domain = RegularGrid((0,5),dx)
# is a regular grid of [0,5] with spacing dx. But now we have to start talking about boundaries. It's the domain that knows what boundaries it has, and thus what has to be specified. So to the domain we need to add boundary functions.

# Boundary conditions are specified by types <:AbstractBoundaryCondition. So we could have:

lbc = Dirichlet(0)
rbc = Neumann(g(t,x))
# Now we'd package that together:

space = Space(domain,lbc,rbc)
# Those domains are compatible with those boundary conditions, so it works without error. Once we have a space, our Heat Equation is specified via:

u' = D(t,x,u)*A*u + f(t,x,u)
# so we specify:

tspan = (0,15)
prob = HeatProblem(D,f,space,tspan)