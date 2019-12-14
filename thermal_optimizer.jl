using DifferentialEquations
using LinearAlgebra
using Plots
using ForwardDiff
using ProgressMeter

# Thermal Model

function generateStencil!(A, N, Δx)
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

# Discretize in time and space
Δx = 1
x = Δx:Δx:100-Δx # Solve only for the interior: the endpoints are known to be zero!

Δt = 0.1
t = Δt:Δt:1000-Δt # Solve only for the interior: the endpoints are known to be zero!

N = length(x)
A = zeros(N,N)
generateStencil!(A, N, Δx)

# f(u, p, t) = zeros(size(u)) 
f(u, p ,t) = gen_u0(p) 
# u0_func(x) = sin.(2π * x)

# Given x coord of components, give initial conditions that are equiv to a gaussian around that coordinate
function gen_u0(positions)
    out_vec = zeros(size(x))
    for pos in positions
        out_vec += gaussian.(x, pos, 5.0)
    end
    out_vec
end


function gaussian(x, mu, sig)
    return exp(-(x - mu)^2.0 / (2 * sig^2.0)) 
end

function box(x, pos)
    sz = 5.0
    if pos-sz <= x <= pos+sz 
        return 1.0
    else 
        return 0.0
    end
end


function heat_eq(u, p, t)
    du = A * u + f(u,p,t)
    du 
end


# Predict
function predict(positions)
    # Simply roll this forward in time and we get our solution
    tspan = (0.0, 1.0)
    # @show typeof(positions)
    # u₀ = gen_u0(positions)
    u₀ = zeros(size(x))
    # @show typeof(u₀)
    prob = ODEProblem(heat_eq, u₀, tspan, positions)
    sol = solve(prob)
    # @show typeof(sol[end])

    # display("Finished ODE Solve")
    sol
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


# Gradient Descent: Optimize initial positions
function optimize(positions, η=0.1)
    
    @showprogress for idx in 1:1000
        grads = ForwardDiff.gradient(s -> loss(s), positions) # Magic
        positions .-= η*grads
        # display(positions)
        # display(loss(positions))

        if loss(positions) < 0.1
            break
        end
    end

    positions
end

function plot_loss_surf()
    loss_surf = zeros((90, 90))
    for x_1 in 5.0:94.0
        for x_2 in 5.0:94.0
            loss_surf[Int(x_1) - 4, Int(x_2) - 4] = loss([x_1, x_2])
        end
    end

    # sol_f(x, y) = loss_surf[x, y]
    plot(loss_surf, st=:heatmap, color=:viridis)

end
    
# optimize([25.0, 30.0])

function plt_sol(sol)

    sol_f(t, x) = sol[t, x]
    plot(1:length(sol[end]), 1:length(sol.t), sol_f, st=:surface, color=:viridis)

end