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

function qubit_dynamics!(ẋ,x,ü)
      u̇ = x[5]
      u = x[6]
      ẋ .= [(-iSx - u.*iSz)*x[1:4]; ü; u̇; u]
end

n = 7 #state dimension
m = 1 #inut dimension

dt = 0.01 #time step
N = 101 #number of knot points

model = Model(qubit_dynamics!,n,m)
model_d = rk4(model)

x0 = [1.0, 0, 0, 0, 0, 0, 0] #initial state
xf = [1/sqrt(2), 0, 1/sqrt(2), 0, 0, 0, 0] #desired final state

u0 = [randn(m) for k = 1:N-1] #random initial guess for control inputs

#Set up quadratic objective function
Q = 0.001*Diagonal(I,n)
R = 10.0*Diagonal(I,m)
Qf = 1.0*Diagonal(I,n)
obj = LQRObjective(Q,R,Qf,xf,N);

#Set up input constriants
goal = goal_constraint(xf) # terminal constraint
constraints = Constraints(N) # define constraints at each time step
constraints[N] += goal;

prob = Problem(model_d, obj, constraints=constraints, x0=x0, xf=xf, N=N, dt=dt)
initial_controls!(prob, u0)

solver = solve!(prob, ALTROSolverOptions{Float64}())

X = fill(Float64[], N)
U = zeros(N)
U̇ = zeros(N)
for k = 1:N
      X[k] = prob.X[k][1:4]
      U[k] = prob.X[k][6]
      U̇[k] = prob.X[k][5]
end

plot(X,xlabel="time step",title="State Trajectory",label=["Re(x1)" "Im(x1)" "Re(x2)" "Im(x2)"])
plot(U,xlabel="time step",title="Control Trajectory",label="u")
plot(U̇,xlabel="time step",title="Control Trajectory",label="udot")
