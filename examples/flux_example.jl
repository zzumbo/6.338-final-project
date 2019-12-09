#!/usr/bin/julia

using Flux

W = rand(2, 5)
b = rand(2)

predict(x) = (W * x) .+ b
loss(x, y) = sum((predict(x) .- y).^2)

x, y = rand(5), rand(2) # Dummy data
println("Data: ")
println(x, y)

l = loss(x, y) # ~ 3

θ = Flux.Params([W, b])
println("\nInitial params: ", θ)

grads = gradient(() -> loss(x, y), θ)

##########################################

# η = 0.1 # Learning Rate
opt = Descent(0.1) # Gradient descent with learning rate 0.1

for i in 1:10 
  for p in (W, b)
    # Flux.Optimise.update!(p, -η * grads[p]) # Can specify your own optimizer
    Flux.Optimise.update!(opt, p, grads[p])
  end
  println("Loss: ", loss(x, y))
end

println("\nOptimized params: ", θ)
println("Loss: ", loss(x, y))
