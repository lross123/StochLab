using Random
using CSV
using DataFrames
using StatsBase
using FilePathsBase

# In this script, we generate the data required to create Figure 2(a)-(d).

# Get the path of the current script.
path = dirname(@__FILE__)

# Init(): Set up the initial conditions for a stochastic model involving two subpopulations.
function init(W,H,Δ,x1_0,d1_0)
   #------------------------------------------------------------------------------------------------------------------
    # Input:
    # W: (number) The width of the lattice.
    # H: (number) The height of the lattice.
    # Δ: (number) The lattice size.
    # x1_0: (list) List of tuples, each tuple contains the location of an agent from subpopulation 1 initially spread from.
    # d1_0: (vector) List of numbers, each number represents the initial occupancy of an agent from subpopulation 1 at the associated location in x1_0.
    # Output:
    # A: (matrix) Lattice in matrix form, A[j,i] is associated with site (J+1-j,i) of the lattice.
    # N: (number) Total number of agents on the lattice.
    #------------------------------------------------------------------------------------------------------------------
    # Number of agents in subpopulation 1.
    N1 = 0; 
    # Generate the empty lattice matrix based on W and H.
    I = Int(W/Δ + 1); J = Int(H/Δ + 1); 
    A = zeros(J,I);
    # Place agents from subpopulation 1 on the lattice based on x1_0 and d1_0.
    # Point to the first column.
    index = 1
    for location in x1_0
        # Transform locations into lattice site indices.
        i_low = Int(location[1]/Δ + 1); i_up = Int(location[2]/Δ + 1);
        # Get the number of agents in each column from column "i_low" to "i_up" from d1_0.
        C1_i = Int(d1_0[index] * J)
        # Point to the next column.
        index = index + 1
        # Place agents on the ith column of the lattice based on d1_0.
        for i = i_low:i_up
            # Place C1_i agents randomly on the ith column.
            js = sample(1:J, C1_i, replace=false)
            for j in js
                A[j,i] = 1
                N1 = N1 + 1
            end
        end
    end
    N = N1
    return A,N
end 

# ctrfer(): Transform the j-th row and i-th column of the lattice matrix A into the lattice site (i,j).
function ctrfer(i,j,J)
    #------------------------------------------------------------------------------------------------------------------
    # Input:
    # i: (number) i-th column of the lattice matrix A.
    # j: (number) j-th row of the lattice matrix A.
    # J: (number) Number of rows in the lattice.
    # Output:
    # (site_i, site_j): (tuple) Lattice site (site_i, site_j) corresponding to the jth row and ith column of the lattice matrix A.
    #------------------------------------------------------------------------------------------------------------------
    site_i = i
    site_j = J+1 - j
    return (site_i,site_j)
end

# Indices(): Extract the indices of each agent in the lattice matrix A and store them in a dataframe.
# Here, an agent with index=(i,j) is equivalent to an agent at site (i,j).
function Indices(A)  
#------------------------------------------------------------------------------------------------------------------
    # Input:
    # A: (matrix) Lattice in matrix form, A[j,i] is associated with site (J+1-j,i) of the lattice.
    # Output:
    # index: (dataframe) index.i stores all the i-indices of agents in subpopulation 1 on the lattice, and the corresponding j-indices are stored in index.j.
    #------------------------------------------------------------------------------------------------------------------
    (J,I) = size(A)
    # Indices for subpopulation 1.
    indices = []
    # Extract the indices of each agent on the lattice and store them as a list of tuples.
    for j in 1:J
        for i in 1:I
            if A[j,i] == 1
                # The selected site is occupied by an agent.
                site = ctrfer(i,j,J)
                push!(indices, site)
            end
        end
    end
    # Store i and j indices for subpopulation 1 in separate vectors.
    i_indices = [index[1] for index in indices]
    j_indices = [index[2] for index in indices]
    # Store i and j indices for subpopulation 1 in dataframes index.
    index = DataFrame(i = vec(i_indices),j = vec(j_indices))
    return index
end

# Count(): Simulate the count data for subpopulations 1.
function Count(index,W,Δ)
#------------------------------------------------------------------------------------------------------------------
    # Input:
    # index: (dataframe) index.i stores all the i-indices of agents in subpopulation 1 on the lattice, and the corresponding j-indices are stored in index.j at the same location.
    # Δ: (number) The lattice size.
    # Output:
    # C_1: (vector) Column count data for subpopulation 1.
    #------------------------------------------------------------------------------------------------------------------
    # Generate empty C_1.
    I = Int(W/Δ + 1);
    C_1 = zeros(I,1)
    # Count the number of agents in subpopulation 1 in each column.
    for i in index.i
        C_1[i] = C_1[i]+1
    end
    return C_1
end

# realisation(): Update the lattice matrix A from current time t to time t + τ.
function realisation(A,P1,ρ1,I,J,N)
    #------------------------------------------------------------------------------------------------------------------
    # Input:
    # A: (matrix) Current lattice in matrix form, A[j,i] is associated with site (J+1-j,i) of the lattice.
    # P1: (number) Motility probability for agents in subpopulation 1.
    # ρ1: (number) Bias parameter for agents in subpopulation 1.
    # I: (number) Number of columns in the lattice.
    # J: (number) Number of rows in the lattice.
    # N: (number) Total number of agents on the lattice.
    # Output:
    # A: (matrix) Updated lattice in matrix form, A[j,i] is associated with site (J+1-j,i) of the lattice.
    #------------------------------------------------------------------------------------------------------------------
    # Probability to move to the right for subpopulations 1.
    prob_right = (1 + ρ1) / 4
    # Number of selected agents from subpopulations 1.
    n1 = 0;
    # Random sequential update - select until we have a total of N agents.
    while n1 < N
        # Select a random ith column and jth row of the lattice matrix.
        i = rand(1:I); j = rand(1:J)
        if A[j,i] == 1 
            # An agent from subpopulation 1 is selected.
            n1 = n1 + 1
            if rand() <= P1
                # If the agent moves, decide the direction of movement.
                prob = rand()
                if prob <= 1/4   # move up
                    if j==1 # at boundary-periodic boundary condition
                        if A[J,i] == 0 
                            A[J,i] = 1
                            A[j,i] = 0
                        end
                    else # not at boundary-move as normal
                        if A[j-1,i] == 0 
                            A[j-1,i] = 1
                            A[j,i] = 0
                        end
                    end
                elseif prob <= 1/2 # move down
                    if j==J # at boundary-periodic boundary condition
                        if A[1,i] == 0 
                            A[1,i] = 1
                            A[j,i] = 0
                        end
                    else # not at boundary-move as normal
                        if A[j+1,i] == 0 
                            A[j+1,i] = 1
                            A[j,i] = 0
                        end
                    end
                elseif prob <= 1/2 + prob_right # move right
                    if i != I # not at boundary-move as normal
                        if A[j,i+1] == 0
                            A[j,i+1] = 1
                            A[j,i] = 0
                        end
                    end
                else # move left
                    if i != 1 # not at boundary-move as normal
                        if A[j,i-1] == 0
                            A[j,i-1] = 1
                            A[j,i] = 0
                        end
                    end
                end
            end
        end
    end
    return A
end

# discrete_simulation(): Update the lattice matrix A from time 0 to time t.
function discrete_simulation(A,N,P1,ρ1,W,H,Δ,τ,t)
    #------------------------------------------------------------------------------------------------------------------
    # Input:
    # A: (matrix) Lattice in matrix form at initial condition, A[j,i] is associated with site (J+1-j,i) of the lattice.
    # N: (number) Total number of agents on the lattice.
    # P1: (number) Motility probability for agents in subpopulation 1.
    # ρ1: (number) Bias parameter for agents in subpopulation 1.
    # W: (number) The width of the lattice.
    # H: (number) The height of the lattice.
    # Δ: (number) The lattice size.
    # τ: (number) The discrete time step duration.
    # t: (number) Simulation time.
    # Output:
    # A: (matrix) Updated lattice in matrix form at time t, A[j,i] is associated with site (J+1-j,i) of the lattice.
    #------------------------------------------------------------------------------------------------------------------
    # Calculate the number of rows and columns of the lattice.
    I = Int(W/Δ + 1); J = Int(H/Δ + 1); 
    # Update the lattice matrix with a discrete time step τ.
    for i = τ:τ:t
        A = realisation(A,P1,ρ1,I,J,N)
    end 
    return A
end

# Lattice size.
W = 199; H = 19; Δ = 1
# Time step duration.
τ=1
# Discrete model parameters.
P1 = 1; ρ1 = 0.1
# Simuation time.
t = 300

# Inital condition
#---------------------------
# Agents in subpopulation 1 initially fully occupy sites at 10<=i<=40, 
# which correspond to Cartesian coordinate 9<=x<=39. 
x_0 = [(9,39)]
d_0 = [1]

# Generate the initial condition.
A0,N= init(W,H,Δ,x_0,d_0)

# Generate indices for subpopulations 1 at the initial condition.
index_0 = Indices(A0) 

# Save indices for subpopulations 1 at the initial condition.
CSV.write("$path\\index_0.csv", index_0)


# Generate count data for subpopulations 1 at the initial condition.
C_1_0 = Count(index_0,W,Δ)

# Save the count data for subpopulations 1 at the initial condition.
datas_0 = DataFrame(a = vec(C_1_0))
CSV.write("$path\\data_0.csv", datas_0)

# Run the discrete simulation from time 0 to time t.
A=discrete_simulation(A0,N,P1,ρ1,W,H,Δ,τ,t)

# Generate indices for each agent from subpopulations 1 at time t.
index = Indices(A) 

# Save the indices for subpopulations 1 at time t.
CSV.write("$path\\index.csv", index)


# Generate count data for subpopulations 1 at time t.
C_1 = Count(index,W,Δ)

# Save the count data for subpopulations 1 at time t.
datas = DataFrame(a = vec(C_1))
CSV.write("$path\\data.csv", datas) 

