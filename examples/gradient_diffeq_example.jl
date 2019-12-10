using OrdinaryDiffEq, ParameterizedFunctions, ForwardDiff

f = @ode_def begin
  dx = a*x - b*x*y
  dy = -c*y + x*y
end a b c

p = [1.5,1.0,3.0]
prob = ODEProblem(f,[1.0;1.0],(0.0,10.0),p)
t = 0.0:0.5:10.0

function G(p)
  tmp_prob = remake(prob,u0=convert.(eltype(p),prob.u0),p=p)
  sol = solve(tmp_prob,Vern9(),abstol=1e-14,reltol=1e-14,saveat=t)
  A = convert(Array,sol)
  sum(((1 .- A).^2)./2)
end
# G([1.5,1.0,3.0])
# res2 = ForwardDiff.gradient(G,[1.5,1.0,3.0])

@info("Running forwarddiff")
# G([1.5,1.0,3.0])
es2 = ForwardDiff.gradient(G,[1.5,1.0,3.0])
# using Zygote
# Zygote.gradient(G,[1.5,1.0,3.0])