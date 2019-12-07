using LinearAlgebra
using TrajectoryOptimization
using Plots

#Pauli spin matrices
i2 = [0 -1; 1 0] #2x2 unit imaginary matrix
Sx = [zeros(2,2) I; I zeros(2,2)]
Sy = [zeros(2,2) -i2; i2 zeros(2,2)]
Sz = [I zeros(2,2); zeros(2,2) -I]
iSx = [zeros(2,2) i2; i2 zeros(2,2)]
iSy = [zeros(2,2) I; -I zeros(2,2)]
iSz = [i2 zeros(2,2); zeros(2,2) -i2]

function qubit_dynamics!(ẋ,x,u)
      ẋ .= (Sx + u.*Sz)*x
end

n = 4 #state dimension
m = 1 #inut dimension

model = Model(qubit_dynamics!,n,m)
model_d = rk4(model)

dt = 0.01 #time step
N = 101 #number of knot points

x0 = [1.0, 0, 0, 0] #initial state
xf = [1/sqrt(2), 0, 1/sqrt(2), 0] #desired final state

u0 = [randn(m) for k = 1:N-1] #random initial guess for control inputs

#Set up quadratic objective function
Q = 1.0*Diagonal(I,n)
R = 0.1*Diagonal(I,m)
Qf = 10.0*Diagonal(I,n)
obj = LQRObjective(Q,R,Qf,xf,N)

prob = Problem(model_d, obj, x0=x0, xf=xf, N=N, dt=dt)
initial_controls!(prob, u0)

solver = solve!(prob, iLQRSolverOptions{Float64}(square_root=true, cost_tolerance=1.0e-8, verbose=true))

plotly()
plot(prob.X,xlabel="time step",title="State Trajectory",label=["Re(x1)" "Im(x1)" "Re(x2)" "Im(x2)"])
plot(prob.U,xlabel="time step",title="Control Trajectory")
