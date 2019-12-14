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

f(u, p, t) = gen_forcing_2D(p) 

function gen_forcing_2D(positions)
    out_vec = zeros(size(x))
    out_vec = convert.(eltype(positions),out_vec)
    for pos in positions
        out_vec += gaussian_2D.(x, pos, 5.0)
    end
    out_vec
end

function heat_eq(u, p, t)
    du = double_partial_2D(u) + f(u,p,t)
    du 
end

function gaussian(x, mu, sig)
    return exp(-(x - mu)^2.0 / (2 * sig^2.0)) 
end

function gaussian_2D(x, y, x₀, y₀, σ₁, σ₂)
    return exp(-( (x-x₀)^2.0/(2.0*σ₁^2) + (y-y₀)^2.0/(2.0*σ₂^2.0) ) )
end

function create_2D_gaussian_matrix(xs,ys,positions)
    out = zeros(size(xs)[1],size(ys)[1])
    out = convert.(eltype(positions[1]),out)
    x_idx = 1
    y_idx = 1

    for pos in positions
        x_p, y_p = pos
        for x in xs
            for y in ys
                out[x_idx, y_idx] += gaussian_2D(x,y,x_p,y_p,5.0,5.0)
                y_idx += 1
            end
            y_idx = 1
            x_idx += 1
        end
        x_idx = 1

    end
    out
end

function plot_gaussian()
    positions = [[20.0,30.0],[40.0,50.0]]
    xs = 1.0:100.0
    ys = 1.0:100.0
    gm = create_2D_gaussian_matrix(xs,ys,positions)
    gm_f(x, y) = gm[x, y]
    display(plot(1:length(xs), 1:length(ys), gm_f, st=:surface, color=:viridis))
end