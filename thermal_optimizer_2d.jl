using DifferentialEquations
using LinearAlgebra
using Plots
using ForwardDiff
using ProgressMeter
using DSP

N = 4
A = zeros(N,N)

x_size = 10
x = reshape(collect(1.0:100.0),(10,10))

stencil = [0.0  1.0 0.0; 
           1.0 -4.0 1.0; 
           0.0  1.0 0.0]

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
      dx[i] = 1*x[i-1] - 2*x[i] + 1*x[i+1]
    end
    dx[1] = -(2*x[1] + -1*x[2])
    dx[end] = -(-1*x[end-1] + 2*x[end])
end

function double_partial_2D(x)
    padded_du = conv2(x, stencil)
    du = padded_du[2:end-1,2:end-1]
    du
end