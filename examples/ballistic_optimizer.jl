using DifferentialEquations
using LinearAlgebra
using Plots
using ForwardDiff

# 2D Ball Example
function physics(u, p, t)
    # x, dx = u[1, 3]
    # y, dy = u[2, 4]

    # g = gravity, b = drag
    g, b = p[1], p[2]

    dx = u[2]
    ddx = b * u[1]^2 
    dy = u[4]
    ddy = -g - b * u[3]^2

    [dx, ddx, dy, ddy]
end

# Callback
condition_cont(u,t,integrator) = u[3]
condition(u,t,integrator) = u[3]<=0
affect!(integrator) = terminate!(integrator)
cb = DiscreteCallback(condition,affect!)
# cb = ContinuousCallback(condition_cont,nothing,affect!)


#Lets wrap this into a "predict" method
function predict(vel)
    # Simply roll this forward in time and we get our solution
    tspan = (0.0, 10.0)
    p = [9.8, 0.0]

    start_state = [0.0, vel[1], 10.0, vel[2]]
    prob = ODEProblem(physics, start_state, tspan, p)
    sol = solve(prob, callback=cb)

    sol
end

## Now let's try to optimize the system to hit a particular point
# want point (1.0, 0)
function loss(vel)
    simulation = predict(vel)
    final_pos = simulation[[1, 3], end]
    l = (final_pos[1] - 1.0)^2 + final_pos[2]^2 # x -> 1.0, y -> 0
    return l
end

# Plots loss surface over velocities in x and y
function plot_loss_surf()
    v_x = 0.0:0.01:3 
    v_y = 0.0:0.01:2

    loss_mat = zeros(length(v_x), length(v_y))
    for x in 1:length(v_x)
        for y in 1:length(v_y)
            loss_mat[x, y] = loss([v_x[x], v_y[y]])
        end
    end

    # sol_f(x, y) = loss_mat[x, y]
    # plot(1:length(v_x), 1:length(v_y), sol_f, st=:surf, color=:viridis)
    plot(loss_mat, st=:heatmap, color=:viridis)

end

# Plots loss over just x velocities
function plot_loss_x()

    v_x = 0.05:0.01:3 

    loss_mat = zeros(length(v_x))
    for x in 1:length(v_x)
        loss_mat[x] = loss([v_x[x], 0.0])
    end

    plot(v_x, loss_mat)

end


# Optimize the xy velocities needed
function optimize(vel_opt, η)
    for idx in 1:1000
        grads = ForwardDiff.gradient(s -> loss(s), vel_opt) # Magic
        vel_opt .-= η*grads

        if loss(vel_opt) < 0.1
            break
        end
    end

    vel_opt
end

# Example run of optimize
p = [1.0, 0.4]
init_sol = predict(p); 
v = optimize(p, 0.01); 
final_sol = predict(v);

plot(init_sol, vars=(1, 3), label="Initial"); plot!(final_sol, vars=(1,3), label="Optimized to land at x=1.0, y=0.0")