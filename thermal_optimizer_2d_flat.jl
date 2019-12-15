using DifferentialEquations
using LinearAlgebra
using Plots
using ForwardDiff
using ProgressMeter
using DSP
using Serialization

N = 4
A = zeros(N,N)

xs = 1.0:30.0
ys = 1.0:30.0
x = zeros(length(xs),length(ys))
positions = [10.0,10.0,13.0,13.0] # Two components, one at (20.0,30.0) and one at (30.0,40.0)

stencil = [0.0  1.0 0.0; 
           1.0 -4.0 1.0; 
           0.0  1.0 0.0]

# generate_u_prime(x, dx)

# function generate1DStencil!(A, N, Δx)
#     # Generate the centered difference matrix manually
#     for i in 1:N, j in 1:N

#     if abs(i-j)<=1
#         (A[i,j]+=1)
#     end
#     if i==j
#         (A[i,j]-=3)
#     end

#     end

#     A = A/(Δx^2)
# end

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

x = reshape(collect(1.0:16.0),(4,4))

function double_partial_2D_manual(x)
    nx = zeros(size(x))
    nx = convert.(eltype(x[1,1]),nx)

    for i=1:size(x,1)
        for j=1:size(x,2)
            nx[i,j] += conv_helper(i,j,x,stencil)
        end
    end
    nx
end

function conv_helper(i,j,x,stencil)
    val = 0
    for dx=-1:1
        for dy=-1:1
            ni = i+dx
            nj = j+dy
            if 1 <= ni <= size(x,1) && 1 <= nj <= size(x,2)
                val += x[ni,nj] * stencil[dx+2,dy+2]
            end
        end
    end
    val
end

f(u, p, t) = gen_forcing_2D(p) 

function gen_forcing_2D(positions)
    out_vec = create_2D_gaussian_matrix(xs,ys,positions)
    out_vec
end

function heat_eq(u, p, t)
    du = double_partial_2D_manual(u) + f(u,p,t)
    du 
end

function gaussian(x, mu, sig)
    return exp(-(x - mu)^2.0 / (2 * sig^2.0)) 
end

function gaussian_2D(loc, mu, sig)
    x, y = loc
    x₀, y₀ = mu
    σ₁, σ₂ = sig
    return gaussian(x, x₀, σ₁) * gaussian(y, y₀, σ₂)
end

function gaussian_2D_multipos(loc, positions, sig)
    x, y = loc
    σ₁, σ₂ = sig
    val = 0
    for i=1:2:length(positions)
        x₀, y₀ = positions[i], positions[i+1]
        val += gaussian(x, x₀, σ₁) * gaussian(y, y₀, σ₂)
    end
    val
end

# Loss: Given a list of positions, what is the cost / loss of the 
function loss(positions)
    simulation = predict(positions)
    # display(simulation)
    final_state = simulation[end]
    # display(size(final_state)[1])
    # display(final_state)
    return norm(final_state, 2)
end

function predict(positions)
    # Simply roll this forward in time and we get our solution
    tspan = (0.0, 1.0)
    u₀ = zeros(size(xs,1),size(ys,1))
    u_dual = convert.(eltype(positions[1]),u₀)
    prob = ODEProblem(heat_eq, u_dual, tspan, positions)
    sol = solve(prob)
    # display("Finished ODE Solve")
    sol
end

# function create_2D_gaussian_matrix_orig(xs,ys,positions)
#     out = zeros(size(xs)[1],size(ys)[1])
#     out = convert.(eltype(positions[1]),out)
#     x_idx = 1
#     y_idx = 1

#     for pos in positions
#         x_p, y_p = pos
#         for x in xs
#             for y in ys
#                 out[x_idx, y_idx] += gaussian_2D([x,y],[x_p,y_p],[5.0,5.0])
#                 y_idx += 1
#             end
#             y_idx = 1
#             x_idx += 1
#         end
#         x_idx = 1

#     end
#     out
# end


function create_2D_gaussian_matrix(xs,ys,positions)
    xsz = length(xs)
    ysz = length(ys)
    out = zeros(xsz,ysz)
    out = convert.(eltype(positions[1]),out)


    for i=1:xsz
        for j=1:ysz
            x, y = xs[i], ys[j]
            out[i, j] += gaussian_2D_multipos([x,y],positions,[5.0,5.0])
        end
    end

    out
end

function plot_gaussian()
    positions = [[20.0,30.0],[40.0,50.0]]
    xs = 1.0:100.0
    ys = 1.0:100.0
    gm = create_2D_gaussian_matrix(xs,ys,positions)
    gm_f(x, y) = gm[x, y]
    display(plot(1:length(xs), 1:length(ys), gm_f, st=:heatmap, color=:viridis))
end

function plot_heatmap(data)
    display(plot(1:size(data,1), 1:size(data,2), data, st=:heatmap, color=:viridis))
end

function plot_gaussian_2()
    positions = [[20.0,30.0],[40.0,50.0],[66.0,3.3]]
    xs = 1.0:0.5:100.0
    ys = 1.0:0.5:100.0
    gf = (x,y) -> gaussian_2D_multipos([x,y],positions,[5.0,5.0])
    display(plot(xs, ys, gf, st=:heatmap, color=:viridis))
end

function optimize_and_save_thermal(positions, η=0.1)
    solns = []
    append!(solns, [deepcopy(positions)])
    @showprogress for idx in 1:1000
        grads = ForwardDiff.gradient(loss, positions) # Magic
        positions .-= η*grads
        # display(positions)
        # display(loss(positions))

        if loss(positions) < 0.1
            break
        end

        append!(solns, [deepcopy(positions)])

    end

    solns
end


function test_solns()
    init_pos = [10.0,10.0,13.0,13.0]
    ps = optimize_and_save_thermal(init_pos)
    serialize("positions.bin", ps)

    # @gif for positions in ps
    #     gm = create_2D_gaussian_matrix(xs,ys,positions)
    #     plot(1:size(gm,1), 1:size(gm,2), gm, st=:heatmap, color=:viridis)
    # end

end

function plot_intermediates()
    ps = deserialize("positions.bin")
    ints = []
    for j=1:100:1001 
        append!(ints, [ps[j]])
    end
    
    i = 0
    for positions in ints
        gm = create_2D_gaussian_matrix(xs,ys,positions)
        display(plot(1:size(gm,1), 1:size(gm,2), gm, 
                st=:heatmap, 
                color=:viridis, 
                title="Thermal Optimizer Iteration "*string(i*100), 
                xlabel="x position (meters)", 
                ylabel="y position (meters)"))
        savefig("final_project/plots/thermal_2D_plots/thermal_2D_plot_"*string(i))
        i += 1
    end
end
