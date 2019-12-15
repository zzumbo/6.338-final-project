using Plots
using DifferentialEquations

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

# Semi-linear heat equation: $u_t = \delta u + f$
f(t, u)  = zeros(size(u)) #ones(size(x,1)) - .5u 
u0_func(x) = if (0.4 < x < 0.6) 1.0 else 0.0 end

# Discretize in time and space
Δx = 0.01
x = Δx:Δx:1-Δx # Solve only for the interior: the endpoints are known to be zero!

Δt = 0.1
t = Δt:Δt:1-Δt # Solve only for the interior: the endpoints are known to be zero!

N = length(x)
A = zeros(N,N)
generateStencil!(A, N, Δx)
u0 = u0_func.(x)


function heat_eq(u, p, t)
    du = A * u + f(t, u)
    du 
end

function solve_prob(tmax)
    prob = ODEProblem(heat_eq, u0, (Δt, tmax - Δt))
    t = Δt:Δt:tmax-Δt
    sol = solve(prob, tstops=t)

    sol_f(t, x) = sol[t, x]
    plot(1:length(x), 1:length(t), (t, x) -> sol[t, x], 
        title="Thermal Evolution of Box centered at 0.5",
        xlabel="Position Bins", 
        ylabel="Time Bins",
        st=:heatmap, 
        color=:viridis)
    savefig("plots/heat_example.png")
end