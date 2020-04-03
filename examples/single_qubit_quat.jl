using TrajectoryOptimization
using TrajOptCore
using RobotDynamics
using LinearAlgebra
using StaticArrays
using Rotations
using TrajOptPlots
using LaTeXStrings

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                 Set up Model
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

struct Qubit{D,N} <: LieGroupModel
    function Qubit(nderiv::Int=0, integrate::Bool=false)
        new{nderiv, integrate}()
    end
end

Base.size(::Qubit{D,N}) where {D,N} = 4+D+N,1
Base.position(::Qubit) = SA_F64[0,0,0]
RobotDynamics.orientation(::Qubit, x::AbstractVector) = UnitQuaternion(x[1], x[2], x[3], x[4])

RobotDynamics.state_diff_size(::Qubit{D,N}) where {D,N} = 3+D+N

function RobotDynamics.state_diff(::Qubit{D,N}, x::SVector, x0::SVector) where {D,N}
    q  = UnitQuaternion(x[1], x[2], x[3], x[4])
    q0 = UnitQuaternion(x0[1], x0[2], x0[3], x0[4])
    δq = q ⊖ q0
    ui = 4 .+ SVector{D+N}(1:D+N)
    δu = x[ui] - x0[ui]
    return [δq; δu]
end

function RobotDynamics.state_diff_jacobian!(G, model::Qubit, x::SVector)
    q = orientation(model, x)
    G0 = Rotations.∇differential(q)
    G[1:4, 1:3] .= G0
    for i = 5:length(x)
        G[i,i-1] = 1
    end
    return nothing
end

function RobotDynamics.∇²differential!(∇G, model::Qubit, x::SVector, dx::AbstractVector)
    q  = orientation(model, x)
    dq = Rotations.params(orientation(model, dx))
    # q = UnitQuaternion(x[1], x[2], x[3], x[4])
    # dq = SA[dx[1], dx[2], dx[3], dx[4]]
    ∇G0 = Rotations.∇²differential(q, dq)
    ∇G[1:3, 1:3] .= ∇G0
    return nothing
end

function RobotDynamics.dynamics(model::Qubit{D,N}, x, u) where {D,N}
    #Pauli spin matrices
    iSx = SA[0  0  0 -1;
             0  0  1  0;
             0 -1  0  0;
             1  0  0  0]
    iSx = SA[0  0  0 -1;
             0  0  1  0;
             0 -1  0  0;
             1  0  0  0]
    iSz = SA[0 -1  0  0;
             1  0  0  0;
             0  0  0  1;
             0  0 -1  0]

    q = @SVector [x[1], x[2], x[3], x[4]]
    qd = (-iSx - u.*iSz)*q
    iu = 4 .+ SVector{D}(1:D)
    ud = x[iu] # derivatives of u: [udot, u] (D=2), [u] (D=1)
    udd = pushfirst(ud, u[1]) # add derivative from the control to the front
    if !N
        udd = pop(udd) # remove u if ∫u isn't the last state
    end
    xdot = [qd; udd]
end

function Base.rand(::Qubit{D,N}) where {D,N}
    q = Rotations.params(rand(UnitQuaternion))
    if D+N > 0
        ud = @SVector rand(D+N)
    else
        ud = @SVector zeros(0)
    end
    return [q;ud], @SVector rand(1)
end


# Discretization
dt = 0.01  # time step
N = 101    # nubmer of knot points
tf = dt*(N-1)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                With control integral -> 0
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Set up model
model = Qubit(0,true)
n,m = size(model)
nd = n - 4  # number of extra states in addition to the quaternion
n̄ = RobotDynamics.state_diff_size(model)

# Set initial and final states
q0 = @SVector [1.0,  0, 0,    0]   # initial quaternion
qf = @SVector [1/√2, 0, 1/√2, 0]   # final quaternion
x0 = [q0; @SVector zeros(nd)]      # control and derivatives should start at 0
xf = [qf; @SVector zeros(nd)]      # control and derivatives (+integral) should stop at 0

U0 = [@SVector zeros(m) for k = 1:N-1]  # random initial guess for control inputs

# Objective
Q = Diagonal(@SVector fill(1.0, n))
R = Diagonal(@SVector fill(2.0,m))
Qf = 1000*Q
obj = LQRObjective(Q,R,Qf,xf,N,checks=false)

# penalize control at first and last time step more
costfun = LQRCost(Q,100*R,xf)
obj.cost[1] = costfun
obj.cost[N-1] = costfun

# Solve unconstrained problem
prob = Problem(model, obj, xf, tf, x0=x0, U0=U0)
solver = iLQRSolver(prob)
# benchmark_solve!(solver)
solver.opts.verbose = true
solve!(solver)
err = rotation_angle(UnitQuaternion(states(solver)[end][1:4])\UnitQuaternion(xf[1:4])) # angular error
err = rad2deg(err)
iterations(solver)

t = RobotDynamics.get_times(solver)
plot(t,controls(solver), xlabel="time (s)", ylabel="control", label="")
plot(t,states(solver),5:5, xlabel="time (s)", ylabel="control integral", label="")
states(solver)[end][end]  # final control integral
sum([u[1] for u in controls(solver)])/(N-1) ≈ states(solver)[end][end]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                With 1st order smoothing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Set up model
model = Qubit(1,true)                   # use integral and control the 1st derivative of the actual control
n,m = size(model)
n̄ = RobotDynamics.state_diff_size(model)
nd = n - 4                              # number of extra states in addition to the quaternion

# Set initial and final states
q0 = @SVector [1.0,  0, 0,    0];        # initial quaternion
qf = @SVector [1/√2, 0, 1/√2, 0];        # final quaternion
x0 = [q0; @SVector zeros(nd)];           # control and derivatives should start at 0
xf = [qf; @SVector zeros(nd)];           # control and derivatives (+integral) should stop at 0

U0 = [@SVector randn(m) for k = 1:N-1];  # random initial guess for control inputs
U0 .*= 0.00

# Objective
Qq = @SVector fill(1e-6, 4);             # weight on quaternion
Qu = SA[0.1, 0.01];                      # weight on control and control integral
Q = Diagonal([Qq; Qu]);
R = Diagonal(@SVector fill(1.0,m));      # weight on control derivative
Qfq = @SVector fill(100.0, 4);
Qfu = SA[0.1, 10.0]
Qf = 100*Diagonal([Qfq; Qfu]);
obj = LQRObjective(Q,R,Qf,xf,N,checks=false)

# Solve unconstrained problem
prob = Problem(model, obj, xf, tf, x0=x0, U0=U0)
solver = iLQRSolver(prob)
# benchmark_solve!(solver)
solver.opts.verbose = true
solve!(solver)
err = rotation_angle(UnitQuaternion(states(solver)[end][1:4])\UnitQuaternion(xf[1:4]))
err = rad2deg(err) # angular error (deg)
iterations(solver)

t = RobotDynamics.get_times(solver)
plot(t,controls(solver), xlabel="time (s)", ylabel=L"\dot{u}", label="", size=(400,200))
plot(t,states(solver),5:5, xlabel="time (s)", ylabel=L"u", label="")
plot(t,states(solver),6:6, xlabel="time (s)", ylabel=L"\int u", label="", size=(400,200))
states(solver)[end][end]  # final control integral
sum([x[5] for x in states(solver)])/N


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                With 2nd order smoothing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Set up model
model = Qubit(2,true)                    # use integral and control the 1st derivative of the actual control
n,m = size(model)
nd = n - 4                               # number of extra states in addition to the quaternion

# Set initial and final states
q0 = @SVector [1.0,  0, 0,    0];        # initial quaternion
qf = @SVector [1/√2, 0, 1/√2, 0];        # final quaternion
x0 = [q0; @SVector zeros(nd)];           # control and derivatives should start at 0
xf = [qf; @SVector zeros(nd)];           # control and derivatives (+integral) should stop at 0

U0 = [0 .+ 0.00*@SVector randn(m) for k = 1:N-1];  # random initial guess for control inputs

# Objective
Qq = @SVector fill(1e-6, 4);             # weight on quaternion
Qu = SA[0.5, 0.1, 0.01];                 # weight on du, u, ∫u
Q = Diagonal([Qq; Qu]);
R = Diagonal(@SVector fill(0.5,m));      # weight on control derivative
Qfq = @SVector fill(50.0, 4);
Qfu = SA[0.5, 100, 100.0]
Qf = 100*Diagonal([Qfq; Qfu]);
obj = LQRObjective(Q,R,Qf,xf,N,checks=false)

# Solve unconstrained problem
prob = Problem(model, obj, xf, tf, x0=x0, U0=U0)
solver = iLQRSolver(prob)
benchmark_solve!(solver)
solver.opts.verbose = true
solve!(solver)
err = rotation_angle(UnitQuaternion(states(solver)[end][1:4])\UnitQuaternion(xf[1:4]))
err = rad2deg(err) # angular error (deg)
iterations(solver)

t = RobotDynamics.get_times(solver  )
plot(t,controls(solver), xlabel="time (s)", ylabel=L"\ddot{u}", label="", size=(400,200))
plot(t,states(solver),5:5, xlabel="time (s)", ylabel=L"\dot{u}", label="", size=(400,200))
plot(t,states(solver),6:6, xlabel="time (s)", ylabel=L"u", label="")
plot(t,states(solver),7:7, xlabel="time (s)", ylabel=L"\int u", label="", size=(400,200))
states(solver)[end][end]  # final control integral
sum([x[6] for x in states(solver)])/N
