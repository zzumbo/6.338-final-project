using Plots

# Poisson equation: u_xx = b
# Use centered diff: u_xx = 1/dx^2 (u_i+1 - 2u_i + u_i-1)
true_u(x) = - (1/4π^2) * sin.(2π * x)
u_xx(x) = sin.(2π * x)

Δx = 0.01
x = Δx:Δx:1-Δx # Solve only for the interior: the endpoints are known to be zero!
N = length(x)
B = u_xx(x)
A = zeros(N,N)

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
U = A\B

plot([0;x;1],[0;U;0],label="U")
plot!([0; x; 1], [0; true_u(x); 0], label="Analytical")


