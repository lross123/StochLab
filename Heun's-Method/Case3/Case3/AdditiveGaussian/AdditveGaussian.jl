using Plots, DifferentialEquations
using CSV
using DataFrames
using LaTeXStrings
using Interpolations
using Distributions
using Measures
using NLopt
using FilePathsBase
using  Dierckx

# In this script, we generate the result for additive Gaussian measurement error model for Case 3

# Get the path of the current script
path = dirname(@__FILE__)

# Read count data from the CSV file.
data = DataFrame(CSV.File("$path\\data.csv"));
# Count data for subpopulation 1 and 2 in Case 2.
# Here, data1 corresponds to C_1^{o} and data2 corresponds to C_2^{o}.
data1 = data.a
data2 = data.b
data3 = data.c
# Fixed parameters
# Data is collected at time t = t1.
t1 = 2000
# The PDE is solved on 0<=x<=L with grid spacing δ.
δ=0.5;L=199; J=20

# diff!(): Discretized PDE.
function diff!(d_u, u, p, t)
    # ------------------------------------------------------------------------------------------------------------------
    # Input:
    # d_u: (matrix) d_u[1, :] is dc1/dt， d_u[2, :] is dc2/dt and d_u[3, :] is dc3/dt.
    # u: (matrix)  u[1,n] is c1(x_n)， u[2,n] is c2(x_n) and u[3,n] is c3(x_n)。
    # p: (vector) [D1, D2, D3,δ].
    # t: (number) Time.
    # Output:
    # d_u: (matrix) d_u[1, :] is dc1/dt， d_u[2, :] is dc2/dt and d_u[3, :] is dc3/dt.
    # ------------------------------------------------------------------------------------------------------------------

    (D1,D2,D3,δ) = p
    (S,N) = size(u)

    # Boundary at x = 0
    d_u[1,1] = (D1/(δ^2))*((1-u[1,2]-u[2,2]-u[3,2])*(u[1,2]-u[1,1]) + u[1,2]*(u[3,2]+u[2,2]+u[1,2]-u[3,1]-u[2,1]-u[1,1]))
    d_u[2,1] = (D2/(δ^2))*((1-u[1,2]-u[2,2]-u[3,2])*(u[2,2]-u[2,1]) + u[2,2]*(u[3,2]+u[2,2]+u[1,2]-u[3,1]-u[2,1]-u[1,1]))
    d_u[3,1] = (D3/(δ^2))*((1-u[1,2]-u[2,2]-u[3,2])*(u[3,2]-u[3,1]) + u[3,2]*(u[3,2]+u[2,2]+u[1,2]-u[3,1]-u[2,1]-u[1,1]))

    for n in 2:N-1
        d_u[1,n] = (D1/(2*δ^2))*(2-u[1,n]-u[2,n]-u[3,n]-u[1,n+1]-u[2,n+1]-u[3,n+1])*(u[1,n+1]-u[1,n]) + (D1/(2*δ^2))*(u[1,n]+u[1,n+1])*(u[1,n+1]+u[2,n+1]+u[3,n+1]-u[1,n]-u[2,n]-u[3,n]) - (D1/(2*δ^2))*(2-u[1,n-1]-u[2,n-1]-u[3,n-1]-u[1,n]-u[2,n]-u[3,n])*(u[1,n]-u[1,n-1]) - (D1/(2*δ^2))*(u[1,n]+u[1,n-1])*(u[1,n]+u[2,n]+u[3,n]-u[1,n-1]-u[2,n-1]-u[3,n-1])    
        d_u[2,n] = (D2/(2*δ^2))*(2-u[1,n]-u[2,n]-u[3,n]-u[1,n+1]-u[2,n+1]-u[3,n+1])*(u[2,n+1]-u[2,n]) + (D2/(2*δ^2))*(u[2,n]+u[2,n+1])*(u[1,n+1]+u[2,n+1]+u[3,n+1]-u[1,n]-u[2,n]-u[3,n]) - (D2/(2*δ^2))*(2-u[1,n-1]-u[2,n-1]-u[3,n-1]-u[1,n]-u[2,n]-u[3,n])*(u[2,n]-u[2,n-1]) - (D2/(2*δ^2))*(u[2,n]+u[2,n-1])*(u[1,n]+u[2,n]+u[3,n]-u[1,n-1]-u[2,n-1]-u[3,n-1]) 
        d_u[3,n] = (D3/(2*δ^2))*(2-u[1,n]-u[2,n]-u[3,n]-u[1,n+1]-u[2,n+1]-u[3,n+1])*(u[3,n+1]-u[3,n]) + (D3/(2*δ^2))*(u[3,n]+u[3,n+1])*(u[1,n+1]+u[2,n+1]+u[3,n+1]-u[1,n]-u[2,n]-u[3,n]) - (D3/(2*δ^2))*(2-u[1,n-1]-u[2,n-1]-u[3,n-1]-u[1,n]-u[2,n]-u[3,n])*(u[3,n]-u[3,n-1]) - (D3/(2*δ^2))*(u[3,n]+u[3,n-1])*(u[1,n]+u[2,n]+u[3,n]-u[1,n-1]-u[2,n-1]-u[3,n-1]) 
                      
    end
    
    # Boundary at x = 199
    d_u[1,N] =  (-D1/(δ^2))*((1-u[1,N-1]-u[2,N-1]-u[3,N-1])*(u[1,N]-u[1,N-1]) + u[1,N-1]*(u[3,N]+u[2,N]+u[1,N]-u[3,N-1]-u[2,N-1]-u[1,N-1]))
    d_u[2,N] =  (-D2/(δ^2))*((1-u[1,N-1]-u[2,N-1]-u[3,N-1])*(u[2,N]-u[2,N-1]) + u[2,N-1]*(u[3,N]+u[2,N]+u[1,N]-u[3,N-1]-u[2,N-1]-u[1,N-1]))
    d_u[3,N] =  (-D3/(δ^2))*((1-u[1,N-1]-u[2,N-1]-u[3,N-1])*(u[3,N]-u[3,N-1]) + u[3,N-1]*(u[3,N]+u[2,N]+u[1,N]-u[3,N-1]-u[2,N-1]-u[1,N-1]))

    return d_u
end

# pdesolver(): Solves the ODE in diff!().
function pdesolver(time,D1,D2,D3,δ,L)
    # ------------------------------------------------------------------------------------------------------------------
    # Input:
    # Time: (number) Solve ODE/PDE at t = Time.
    # D1: (number) Diffusivity for agents in subpopulation 1.
    # D2: (number) Diffusivity for agents in subpopulation 2.
    # D3: (number) Diffusivity for agents in subpopulation 3.
    # δ: (number) Grid spacing.
    # L: (number) ODE/PDE is solved on 0 <= x <= L.
    # Output:
    # c1, c2, c3: (vector) Solution of ODE/PDE at time t = Time.
    # ------------------------------------------------------------------------------------------------------------------
    # Construct initial condition for PDE.
    # Total number of mesh points N.
    N = Int(round(L/δ + 1))
    # Initial condition u0.
    u0 = zeros(3,N)
    # Update initial condition u0: c1(x) = 0.5 at 79 <= x <= 119.
    for n=1:N
       xx = (n-1)*δ
       if xx >=79 && xx <= 119
          u0[1,n] = 0.5
       end
    end

    # Update initial condition u0: c2(x) = 0.5 at 79 <= x <= 119.
    for n=1:N
        xx = (n-1)*δ
        if xx >=79 && xx <= 119
           u0[2,n] = 0.5
        end
    end
    
    # Update initial condition u0: c3(x) = 0.25 at 0 <= x < 79 and 119 < x <= 199.
    for n=1:N
        xx = (n-1)*δ
        if xx < 79
           u0[3,n] = 0.25
        end
        if xx > 119
           u0[3,n] = 0.25
        end
    end


    # Return the inital conditon if ODE/PDE is solved at t=0
    if time == 0
        c1_0 = vec(u0[1,:]); c2_0 = vec(u0[2,:]); c3_0 = vec(u0[3,:])
        return c1_0,c2_0,c3_0
    end

    # Solve the PDE using Heun's method at t=Time 
    p=(D1,D2,D3,δ)
    tspan=(0,time)
    prob=ODEProblem(diff!,u0,tspan,p)
    alg=Heun() 
    sol=solve(prob,alg,saveat=time);
    sol = sol[end]
    u1 = sol[1,:];  u2= sol[2,:]; u3= sol[3,:]
    c1 = vec(u1); c2 = vec(u2); c3 = vec(u3)
    return c1,c2,c3
end 


# model(): The continuum model used for Case 3.
function model(t1,a,δ,L)
    #------------------------------------------------------------------------------------------------------------------
    # Input:
    # t1: (number) Solve the PDE at t = t1.
    # a: (vector) Parameter vector.
    # δ: (number) Grid spacing.
    # L: (number) PDE is solved on 0 <= x <= L.
    # Output:
    # c1, c2， c3: (vector) Solution of the PDE at time t = Time.
    #------------------------------------------------------------------------------------------------------------------
    # Solve the PDE.

    c1,c2,c3=pdesolver(t1,a[1],a[2],a[3],δ,L)
    return c1,c2,c3
end

# error(): Log-likelihood function for the Additive Gaussian Measurement Error Model for Case 3.
function error(data1,data2,data3,δ,L,a)
    #------------------------------------------------------------------------------------------------------------------
    # Input:
    # data1: (vector) Count data for subpopulation 1.
    # data2: (vector) Count data for subpopulation 2.
    # data3: (vector) Count data for subpopulation 3.
    # δ: (number) Grid spacing.
    # L: (number) PDE solved on 0 <= x <= L.
    # a: (vector) Parameter vector.
    # Output:
    # e: (number) Log-likelihood for the Additive Gaussian Measurement Error Model (Case 3) with parameter vector a.
    #------------------------------------------------------------------------------------------------------------------
    # Solve the PDE at time t1 with parameter vector a.
    c1,c2,c3=model(t1,a,δ,L);
    # Find c1(x_i, t1) and c2(x_i, t1) for i = 1, 2, 3, ..., 200.
    xlocdata = 0:δ:L
    fc1= linear_interpolation(xlocdata,c1);
    fc2= linear_interpolation(xlocdata,c2);
    fc3= linear_interpolation(xlocdata,c3);
    c1 = fc1(0:1:L); c2 = fc2(0:1:L); c3 = fc3(0:1:L)
    # Estimate the loglikelihood function.
    e=0.0;
    dist1 = Normal(0,a[4]); dist2 = Normal(0,a[5]); dist3 = Normal(0,a[6])
    e=loglikelihood(dist1,(data1./J).-c1) + loglikelihood(dist2,(data2./J).-c2) + loglikelihood(dist3,(data3./J).-c3)
    return e
end
    

# fun(): Evaluate the log-likelihood function as a function of the parameters a.
function fun(a)
    # ------------------------------------------------------------------------------------------------------------------
    # Input:
    # a: (vector) Parameter vector.
    # Output:
    # error(data1, data2, data3, δ, lx, a): (number) Log-likelihood for the Additive Gaussian Measurement Error Model (Case 3) with parameter vector a.
    # ------------------------------------------------------------------------------------------------------------------
    return error(data1,data2,data3,δ,L,a)
end

# optimise(): NLopt routine to maximise the function fun, with parameter estimates θ₀ subject to bound constraints lb, ub
function optimise(fun,θ₀,lb,ub;
    dv = false,
    method = dv ? :LD_LBFGS : :LN_BOBYQA,
)

    if dv || String(method)[2] == 'D'
        tomax = fun
        
    else
        tomax = (θ,∂θ) -> fun(θ)
        
    end
    
    opt = Opt(method,length(θ₀))
    opt.max_objective = tomax
    
    opt.lower_bounds = lb       # Lower bound
    opt.upper_bounds = ub       # Upper bound
    opt.local_optimizer = Opt(:LN_NELDERMEAD, length(θ₀))
    opt.maxtime = 60*60
    res = optimize(opt,θ₀)
    return res[[2,1]]
end

# sample_para(): Perform rejection sampling to select M parameter sets within the 95% log-likelihood threshold.
function sample_para(D1_iv,D2_iv,D3_iv,σ1_iv,σ2_iv,σ3_iv,fmle,M)
    # ------------------------------------------------------------------------------------------------------------------
    # Input:
    # D1_iv,D2_iv,D3_iv,σ1_iv,σ2_iv,σ3_iv: (vector) Intervals for D1, D2, D3, σ1, σ2, and σ3 for rejection sampling.
    # fmle: (number) Log-likelihood at MLE.
    # M: (number) Number of parameter sets required within the 95% log-likelihood threshold.
    # Output:
    # sampledP: (vector) M parameter sets within the 95% log-likelihood threshold.
    # ------------------------------------------------------------------------------------------------------------------
    # Degree of freedom.
    df=6
    # 95% log-likelihood threshold.
    llstar=-quantile(Chisq(df),0.95)/2
    # Rejection sampling.
    # Sampled parameter sets within the 95% log-likelihood threshold.
    sampledP=[]
    # Number of sampled parameter sets within the 95% log-likelihood threshold.
    count = 0
    # Keep sampling until we have M parameter sets within the 95% log-likelihood threshold.
    while count< M
        # Sample a parameter set.
        D1g=rand(Uniform(D1_iv[1],D1_iv[2]))
        σ1g=rand(Uniform(σ1_iv[1],σ1_iv[2]))
        D2g=rand(Uniform(D2_iv[1],D2_iv[2]))
        σ2g=rand(Uniform(σ2_iv[1],σ2_iv[2]))
        D3g=rand(Uniform(D3_iv[1],D3_iv[2]))
        σ3g=rand(Uniform(σ3_iv[1],σ3_iv[2]))
        # Evaluate the log-likelihood function using sampled parameter sets.
        ll = fun([D1g,D2g,D3g,σ1g,σ2g,σ3g])
        if ll-fmle >= llstar
            # Sampled parameter set within the 95% log-likelihood threshold.
            # Update number of sampled parameter sets within the 95% log-likelihood threshold.
            count = count + 1
            # Update sampled parameter sets within the 95% log-likelihood threshold.
            push!(sampledP,(D1g,D2g,D3g,σ1g,σ2g,σ3g))
            display(count)
        end
    end
    return sampledP
end

# predicreal(): Construct prediction interval for data realizations.
function predicreal(lsp,t)
    # ------------------------------------------------------------------------------------------------------------------
    # Input:
    # lsp: (vector) Parameter sets within the 95% log-likelihood threshold.
    # t: (number) Prediction interval constructed at time t.
    # Output:
    # lb1, ub1: (vectors) Lower and upper bounds of prediction interval for subpopulation 1.
    # lb2, ub2: (vectors) Lower and upper bounds of prediction interval for subpopulation 2.
    # lb3, ub3: (vectors) Lower and upper bounds of prediction interval for subpopulation 3.
    # ------------------------------------------------------------------------------------------------------------------
    # Generate empty lower and upper bounds of prediction intervals for subpopulations 1, 2 and 3.
    xx = 0:δ:L
    gp = length(xx) 
    lb1 = ones(gp); lb2 = ones(gp); lb3 = ones(gp);
    ub1 = zeros(gp); ub2 = zeros(gp); ub3 = zeros(gp);
    for i = 1:length(lsp)
        # Solve the PDE with parameter set a.
        a = lsp[i]
        u1,u2,u3 = model(t,[a[1],a[2],a[3]],δ,L)
        p1 = linear_interpolation(xx,vec(u1));
        p2 = linear_interpolation(xx,vec(u2));
        p3 = linear_interpolation(xx,vec(u3));

        # Construct prediction interval for data realizations.
        for j = 1:length(xx)
            ulb1 = (quantile(Normal(p1(xx[j]),a[4]),[.05,.95])[1])
            uub1 = (quantile(Normal(p1(xx[j]),a[4]),[.05,.95])[2])
            ulb2 = (quantile(Normal(p2(xx[j]),a[5]),[.05,.95])[1])
            uub2 = (quantile(Normal(p2(xx[j]),a[5]),[.05,.95])[2])
            ulb3 = (quantile(Normal(p3(xx[j]),a[6]),[.05,.95])[1])
            uub3 = (quantile(Normal(p3(xx[j]),a[6]),[.05,.95])[2])

            if ulb1 < lb1[j] 
                lb1[j] = ulb1
            end

            if ulb2 < lb2[j] 
                lb2[j] = ulb2
            end

            if ulb3 < lb3[j] 
                lb3[j] = ulb3
            end

            if uub1 > ub1[j] 
                ub1[j] = uub1
            end

            if uub2 > ub2[j] 
                ub2[j] = uub2
            end

            if uub3 > ub3[j] 
                ub3[j] = uub3
            end
        end
    end
    return lb1,ub1,lb2,ub2,lb3,ub3
end

#Expected Parameters
P1 = 1; P2 = 0.9; P3 = 0.8
D1 = (P1)/(4); D2 = (P2)/(4); D3 = (P3)/(4);
θ = [D1,D2,D3]

#MLE----------------------------------------------------------------------------
# Inital guess.
D1g = 0.18; D2g = 0.15; D3g = 0.15
σ1g= 0.05; σ2g= 0.11; σ3g= 0.11
θG = [D1g,D2g,D3g,σ1g,σ2g,σ3g] # inital guess
lb=[0.0001,0.01,0.01,0,0,0] # lower bound
ub=[0.4,0.4,0.4,1,1,1] # upper bound
# Call numerical optimization routine to give the vector of parameters xopt, and the maximum loglikelihood fopt.
@time (xopt,fopt)  = optimise(fun,θG,lb,ub)
fmle=fopt
# Print MLE parameters.
D1mle=xopt[1]; 
D2mle=xopt[2]; 
D3mle=xopt[3];
σ1mle=xopt[4];
σ2mle=xopt[5];
σ3mle=xopt[6];
println("D1mle: ", D1mle)
println("D2mle: ", D2mle)
println("D3mle: ", D3mle)
println("σ1mle: ", σ1mle)
println("σ2mle: ", σ2mle)
println("σ3mle: ", σ3mle)
# Solve the PDE with MLE parameters
C1mle,C2mle,C3mle = model(t1,[D1mle,D2mle,D3mle],δ,L)

xmle = 1:δ:200
x = 1:1:200
p1=plot(xmle,C1mle.*J, label=false, linewidth=4, color=:black,margin=10mm,left_margin=35mm)
p1=scatter!(x,data[:,1], label=false, grid=false)
display(p1)
savefig(p1,"$path\\MLE1.pdf")

p2=plot(xmle,C2mle.*J, label=false, linewidth=4, color=:black,margin=10mm,left_margin=35mm)
p2=scatter!(x,data[:,2], label=false, grid=false)
display(p2)
savefig(p2,"$path\\MLE2.pdf")

p3=plot(xmle,C3mle.*J, label=false, linewidth=4, color=:black,margin=10mm,left_margin=35mm)
p3=scatter!(x,data[:,3], label=false, grid=false)
display(p3)
savefig(p3,"$path\\MLE3.pdf")
println("MLE Complete---------------------------------------------------------------")

# Profile D1-------------------------------------------------------------------
df = 1
llstar = -quantile(Chisq(df),0.95)/2 # 95% asymptotic threshold
nptss=40 # points on the profile
D1min=0.2# lower bound for the profile 
D1max=0.35# upper bound for the profile
D1range=LinRange(D1min,D1max,nptss) # vector of D1 values along the profile
nrange=zeros(5,nptss) # matrix to store the nuisance parameters once optimized out
llD1=zeros(nptss) # loglikelihood at each point along the profile
nllD1=zeros(nptss) # normalised loglikelihood at each point along the profile

@time for i in 1:nptss
    function fun1(aa) # function to return loglikelihood by fixing the interest parameter along the profile
        return error(data1,data2,data3,δ,L,[D1range[i],aa[1],aa[2],aa[3],aa[4],aa[5]])
    end
    local lb1=[0.01,0.01,0,0,0] # lower bound 
    local ub1=[0.4,0.4,1,1,1] # upper bound 
    local θG1=[D2mle,D3mle,σ1mle,σ2mle,σ3mle] # initial estimate - take the MLE
    @time local (xo,fo)=optimise(fun1,θG1,lb1,ub1)
    nrange[:,i]=xo[:] # store the nuisance parameters
    llD1[i]=fo[1] # store the loglikelihood 
    display(i)
end
nllD1=llD1.-maximum(llD1); # calculate normalised loglikelihood
# Store the loglikelihood in csv.
df = DataFrame(Value = llD1)
CSV.write("$path\\llD1.csv", df)
# Plot the univariate profile likelihood for D1 and hold.
df = CSV.read("$path\\llD1.csv", DataFrames.DataFrame)
llD1 = df[:, 1]
nllD1=llD1.-maximum(llD1)
#= spl=Spline1D(D1range,llD1.-maximum(llD1),w=ones(length(D1range)),k=3,bc="nearest",s=0.3)
nllD1=evaluate(spl,D1range) =#
s1=plot(D1range,nllD1,lw=6,xlim=(0,0.4),ylim=(-3,0.1),legend=false,tickfontsize=20,labelfontsize=20,margin = 10mm,grid=false,
xticks=([0,0.1,0.2,0.3,0.4], [latexstring("0"),"", latexstring("0.2"),"",latexstring("0.4")]),yticks=([-3,-2,-1,0],[latexstring("-3"),latexstring("-2"),latexstring("-1"),latexstring("0")]),framestyle=:box,left_margin = 15mm)
s1=hline!([llstar],lw=6)
s1=vline!([D1mle],lw=6)

function FindinterceptD1()
    UnivariateD1 = linear_interpolation(D1range,vec(nllD1));
    g(x)=UnivariateD1(x)-llstar
    ϵ=(D1max-D1min)/10^6
    x0=D1mle
    x1=D1min
    x2=(x1+x0)/2;
    while abs(x1-x0) > ϵ
        x2=(x1+x0)/2;
        if g(x0)*g(x2) < 0 
            x1=x2
        else
            x0=x2
        end
    end
    D1D1min = x2
    x0=D1mle
    x1=D1max
    x2=(x1+x0)/2;
    while abs(x1-x0) > ϵ
        x2=(x1+x0)/2;
        if g(x0)*g(x2) < 0 
            x1=x2
        else
            x0=x2
        end
    end
    D1D1max = x2
    return D1D1min,D1D1max
end
D1D1min,D1D1max = FindinterceptD1()
println("D1confidence: ", [D1D1min,D1D1max])
println("Profile D1 Complete-----------------------------------------------------------------------------")

# Profile D2-------------------------------------------------------------------
df = 1
llstar = -quantile(Chisq(df),0.95)/2 # 95% asymptotic threshold
nptss=40 # points on the profile
D2min=0.01 # lower bound for the profile
D2max=0.3 # upper bound for the profile
D2range=LinRange(D2min,D2max,nptss) # vector of D2 values along the profile
nrange=zeros(5,nptss) # matrix to store the nuisance parameters once optimized out
llD2=zeros(nptss) # loglikelihood at each point along the profile
nllD2=zeros(nptss) # normalised loglikelihood at each point along the profile

@time for i in 1:nptss
    function fun2(aa) # function to return loglikelihood by fixing the interest parameter along the profile
        return error(data1,data2,data3,δ,L,[aa[1],D2range[i],aa[2],aa[3],aa[4],aa[5]])
    end
    local lb2=[0.0001,0.0001,0,0,0] # lower bound 
    local ub2=[0.4,0.4,1,1,1]  # upper bound
    local θG2=[D1mle,D3mle,σ1mle,σ2mle,σ3mle] # initial estimate - take the MLE 
    @time local (xo,fo)=optimise(fun2,θG2,lb2,ub2)
    nrange[:,i]=xo[:] # store the nuisance parameters
    llD2[i]=fo[1] # store the loglikelihood 
    display(i)
end
nllD2=llD2.-maximum(llD2); # calculate normalised loglikelihood
# Store the loglikelihood in csv.
df = DataFrame(Value = llD2)
CSV.write("$path\\llD2.csv", df)

function FindinterceptD2()
    UnivariateD2 = linear_interpolation(D2range,vec(nllD2));
    g(x)=UnivariateD2(x)-llstar
    ϵ=(D2max-D2min)/10^6
    x0=D2mle
    x1=D2min
    x2=(x1+x0)/2;
    while abs(x1-x0) > ϵ
        x2=(x1+x0)/2;
        if g(x0)*g(x2) < 0 
            x1=x2
        else
            x0=x2
        end
    end
    D2D2min = x2
    x0=D2mle
    x1=D2max
    x2=(x1+x0)/2;
    while abs(x1-x0) > ϵ
        x2=(x1+x0)/2;
        if g(x0)*g(x2) < 0 
            x1=x2
        else
            x0=x2
        end
    end
    D2D2max = x2
    return D2D2min,D2D2max
end
D2D2min,D2D2max = FindinterceptD2()
println("D2confidence: ", [D2D2min,D2D2max])
println("Profile D2 Complete-----------------------------------------------------------------------------")



# Plot the univariate profile likelihood for D2 with D1 and hold.
df = CSV.read("$path\\llD2.csv", DataFrames.DataFrame)
llD2 = df[:, 1]
nllD2=llD2.-maximum(llD2)
s1=plot!(D2range,nllD2,lw=6,xlim=(0,0.4),ylim=(-3,0.1),legend=false,tickfontsize=20,labelfontsize=20,margin = 10mm,grid=false,
xticks=([0,0.1,0.2,0.3,0.4], [latexstring("0"),"", latexstring("0.2"),"",latexstring("0.4")]),yticks=([-3,-2,-1,0],[latexstring("-3"),latexstring("-2"),latexstring("-1"),latexstring("0")]),framestyle=:box,color=1,line=:dashdot,left_margin = 15mm)
s1=vline!([D2mle],lw=6,color=3, line=:dashdot)
s1=hline!([llstar],lw=6,color=2)

# Profile D3-------------------------------------------------------------------
df = 1
llstar = -quantile(Chisq(df),0.95)/2 # 95% asymptotic threshold
nptss=40 # points on the profile
D3min=0.01 # lower bound for the profile
D3max=0.35 # upper bound for the profile
D3range=LinRange(D3min,D3max,nptss) # vector of D3 values along the profile
nrange=zeros(5,nptss) # matrix to store the nuisance parameters once optimized out
llD3=zeros(nptss) # loglikelihood at each point along the profile
nllD3=zeros(nptss) # normalised loglikelihood at each point along the profile

@time for i in 1:nptss
    function fun3(aa) # function to return loglikelihood by fixing the interest parameter along the profile
        return error(data1,data2,data3,δ,L,[aa[1],aa[2],D3range[i],aa[3],aa[4],aa[5]])
    end
    local lb3=[0.0001,0.0001,0,0,0] # lower bound 
    local ub3=[0.4,0.4,1,1,1]  # upper bound
    local θG3=[D1mle,D2mle,σ1mle,σ2mle,σ3mle] # initial estimate - take the MLE 
    @time local (xo,fo)=optimise(fun3,θG3,lb3,ub3)
    nrange[:,i]=xo[:] # store the nuisance parameters
    llD3[i]=fo[1] # store the loglikelihood 
    display(i)
end
nllD3=llD3.-maximum(llD3); # calculate normalised loglikelihood
# Store the loglikelihood in csv.
df = DataFrame(Value = llD3)
CSV.write("$path\\llD3.csv", df)

function FindinterceptD3()
    UnivariateD3 = linear_interpolation(D3range,vec(nllD3));
    g(x)=UnivariateD3(x)-llstar
    ϵ=(D3max-D3min)/10^6
    x0=D3mle
    x1=D3min
    x2=(x1+x0)/2;
    while abs(x1-x0) > ϵ
        x2=(x1+x0)/2;
        if g(x0)*g(x2) < 0 
            x1=x2
        else
            x0=x2
        end
    end
    D3D3min = x2
    x0=D3mle
    x1=D3max
    x2=(x1+x0)/2;
    while abs(x1-x0) > ϵ
        x2=(x1+x0)/2;
        if g(x0)*g(x2) < 0 
            x1=x2
        else
            x0=x2
        end
    end
    D3D3max = x2
    return D3D3min,D3D3max
end
D3D3min,D3D3max = FindinterceptD3()
println("D3confidence: ", [D3D3min,D3D3max])



# Plot the univariate profile likelihood for D3 with D1 and D2.
df = CSV.read("$path\\llD3.csv", DataFrames.DataFrame)
llD3 = df[:, 1]
nllD3=llD3.-maximum(llD3)
s1=plot!(D3range,nllD3,lw=6,xlim=(0,0.4),ylim=(-3,0.1),legend=false,tickfontsize=20,labelfontsize=20,margin = 10mm,grid=false,
xticks=([0,0.1,0.2,0.3,0.4], [latexstring("0"),"", latexstring("0.2"),"",latexstring("0.4")]),yticks=([-3,-2,-1,0],[latexstring("-3"),latexstring("-2"),latexstring("-1"),latexstring("0")]),framestyle=:box,color=1,line=:dot,left_margin = 15mm)
s1=vline!([D3mle],lw=6,color=3, line=:dot)
s1=hline!([llstar],lw=6,color=2)
s1 = annotate!(0.145, -3.6, text(L"D_{1}, D_{2}, D_{3}", :left, 20))
s1 = annotate!(-0.09, -1.5, text(L"\bar{\ell}_{p}", :left, 20))
savefig(s1,"$path\\FigureS7a.pdf") 
display(s1) 

# Profile σ1-------------------------------------------------------------------
df = 1
llstar = -quantile(Chisq(df),0.95)/2 # 95% asymptotic threshold
nptss=40 # points on the profile
σ1min=0.05 # lower bound for the profile 
σ1max=0.08 # upper bound for the profile  
σ1range=LinRange(σ1min,σ1max,nptss) # vector of σ1 values along the profile
nrange=zeros(5,nptss) # matrix to store the nuisance parameters once optimized out
llσ1=zeros(nptss) # loglikelihood at each point along the profile
nllσ1=zeros(nptss) # normalised loglikelihood at each point along the profile

@time for i in 1:nptss
    function fun5(aa) # function to return loglikelihood by fixing the interest parameter along the profile
        return error(data1,data2,data3,δ,L,[aa[1],aa[2],aa[3],σ1range[i],aa[4],aa[5]])
    end
    local lb5=[0.0001,0.0001,000.1,0,0] # lower bound 
    local ub5=[0.4,0.4,0.4,1,1] # upper bound
    local θG5=[D1mle,D2mle,D3mle,σ2mle,σ3mle] # initial estimate - take the MLE 
    @time local (xo,fo)=optimise(fun5,θG5,lb5,ub5)
    nrange[:,i]=xo[:] # store the nuisance parameters
    llσ1[i]=fo[1] # store the loglikelihood 
    display(i)
end
nllσ1=llσ1.-maximum(llσ1); # calculate normalised loglikelihood
# Store the loglikelihood in csv.
df = DataFrame(Value = llσ1)
CSV.write("$path\\lls1.csv", df)

# Plot the univariate profile likelihood for σ1 and hold.
df = CSV.read("$path\\lls1.csv", DataFrames.DataFrame)
llσ1 = df[:, 1]
nllσ1=llσ1.-maximum(llσ1)
s3=plot(σ1range,nllσ1,lw=6,xlim=(0.05,0.11),ylim=(-3,0.1),legend=false,tickfontsize=20,labelfontsize=20,margin = 10mm,grid=false,
xticks=([0.05,0.08,0.11], [latexstring("0.05"),latexstring("0.08"),latexstring("0.11")]),yticks=([-3,-2,-1,0],[latexstring("-3"),latexstring("-2"),latexstring("-1"),latexstring("0")]),framestyle=:box,left_margin = 15mm)
s3=hline!([llstar],lw=6)
s3=vline!([σ1mle],lw=6)

function Findinterceptσ1()
    Univariateσ1 = linear_interpolation(σ1range,vec(nllσ1));
    g(x)=Univariateσ1(x)-llstar
    ϵ=(σ1max-σ1min)/10^6
    x0=σ1mle
    x1=σ1min
    x2=(x1+x0)/2;
    while abs(x1-x0) > ϵ
        x2=(x1+x0)/2;
        if g(x0)*g(x2) < 0 
            x1=x2
        else
            x0=x2
        end
    end
    σ1σ1min = x2
    x0=σ1mle
    x1=σ1max
    x2=(x1+x0)/2;
    while abs(x1-x0) > ϵ
        x2=(x1+x0)/2;
        if g(x0)*g(x2) < 0 
            x1=x2
        else
            x0=x2
        end
    end
    σ1σ1max = x2
    return σ1σ1min,σ1σ1max
end
σ1σ1min,σ1σ1max = Findinterceptσ1()
println("σ1 confidence: ", [σ1σ1min,σ1σ1max])
println("Profile σ1 Complete-----------------------------------------------------------------------------")

# Profile σ2-------------------------------------------------------------------
df = 1
llstar = -quantile(Chisq(df),0.95)/2 # 95% asymptotic threshold
nptss=40 # points on the profile
σ2min=0.04 # lower bound for the profile  
σ2max=0.08 # upper bound for the profile  
σ2range=LinRange(σ2min,σ2max,nptss) # vector of σ2 values along the profile
nrange=zeros(5,nptss) # matrix to store the nuisance parameters once optimized out
llσ2=zeros(nptss) # loglikelihood at each point along the profile
nllσ2=zeros(nptss) # normalised loglikelihood at each point along the profile

@time for i in 1:nptss
    function fun6(aa) # function to return loglikelihood by fixing the interest parameter along the profile
        return error(data1,data2,data3,δ,L,[aa[1],aa[2],aa[3],aa[4],σ2range[i],aa[5]])
    end
    local lb6=[0.0001,0.0001,0.0001,0,0] # lower bound 
    local ub6=[0.4,0.4,0.4,1,1] # upper bound
    local θG6=[D1mle,D2mle,D3mle,σ1mle,σ3mle]
    @time local (xo,fo)=optimise(fun6,θG6,lb6,ub6) # initial estimate - take the MLE 
    nrange[:,i]=xo[:] # store the nuisance parameters
    llσ2[i]=fo[1] # store the loglikelihood 
    display(i)
end
nllσ2=llσ2.-maximum(llσ2); # calculate normalised loglikelihood
# Store the loglikelihood in csv.
df = DataFrame(Value = llσ2)
CSV.write("$path\\lls2.csv", df)

# Plot the univariate profile likelihood for σ2 with σ1.
df = CSV.read("$path\\lls2.csv", DataFrames.DataFrame)
llσ2 = df[:, 1]
nllσ2=llσ2.-maximum(llσ2)
s3=plot!(σ2range,nllσ2,lw=6,xlim=(0.05,0.11),ylim=(-3,0.1),legend=false,tickfontsize=20,labelfontsize=20,margin = 10mm,grid=false,
xticks=([0.05,0.08,0.11], [latexstring("0.05"),latexstring("0.08"),latexstring("0.11")]),yticks=([-3,-2,-1,0],[latexstring("-3"),latexstring("-2"),latexstring("-1"),latexstring("0")]),framestyle=:box,color=1, line=:dashdot,left_margin = 15mm)
s3=vline!([σ2mle],lw=6,color=3, line=:dashdot)
s3=hline!([llstar],lw=6,color=2)

function Findinterceptσ2()
    Univariateσ2 = linear_interpolation(σ2range,vec(nllσ2));
    g(x)=Univariateσ2(x)-llstar
    ϵ=(σ2max-σ2min)/10^6
    x0=σ2mle
    x1=σ2min
    x2=(x1+x0)/2;
    while abs(x1-x0) > ϵ
        x2=(x1+x0)/2;
        if g(x0)*g(x2) < 0 
            x1=x2
        else
            x0=x2
        end
    end
    σ2σ2min = x2
    x0=σ2mle
    x1=σ2max
    x2=(x1+x0)/2;
    while abs(x1-x0) > ϵ
        x2=(x1+x0)/2;
        if g(x0)*g(x2) < 0 
            x1=x2
        else
            x0=x2
        end
    end
    σ2σ2max = x2
    return σ2σ2min,σ2σ2max
end
σ2σ2min,σ2σ2max = Findinterceptσ2()
println("σ2confidence: ", [σ2σ2min,σ2σ2max])
println("Profile σ2 Complete-----------------------------------------------------------------------------")


# Profile σ3-------------------------------------------------------------------
df = 1
llstar = -quantile(Chisq(df),0.95)/2 # 95% asymptotic threshold
nptss=40 # points on the profile
σ3min=0.06 # lower bound for the profile  
σ3max=0.12 # upper bound for the profile  
σ3range=LinRange(σ3min,σ3max,nptss) # vector of σ2 values along the profile
nrange=zeros(5,nptss) # matrix to store the nuisance parameters once optimized out
llσ3=zeros(nptss) # loglikelihood at each point along the profile
nllσ3=zeros(nptss) # normalised loglikelihood at each point along the profile

@time for i in 1:nptss
    function fun6(aa) # function to return loglikelihood by fixing the interest parameter along the profile
        return error(data1,data2,data3,δ,L,[aa[1],aa[2],aa[3],aa[4],aa[5],σ3range[i]])
    end
    local lb6=[0.0001,0.0001,0.0001,0,0] # lower bound 
    local ub6=[0.4,0.4,0.4,1,1] # upper bound
    local θG6=[D1mle,D2mle,D3mle,σ1mle,σ2mle]
    @time local (xo,fo)=optimise(fun6,θG6,lb6,ub6) # initial estimate - take the MLE 
    nrange[:,i]=xo[:] # store the nuisance parameters
    llσ3[i]=fo[1] # store the loglikelihood 
    display(i)
end
nllσ3=llσ3.-maximum(llσ3); # calculate normalised loglikelihood
# Store the loglikelihood in csv.
df = DataFrame(Value = llσ3)
CSV.write("$path\\lls3.csv", df)

function Findinterceptσ3()
    Univariateσ3 = linear_interpolation(σ3range,vec(nllσ3));
    g(x)=Univariateσ3(x)-llstar
    ϵ=(σ3max-σ3min)/10^6
    x0=σ3mle
    x1=σ3min
    x2=(x1+x0)/2;
    while abs(x1-x0) > ϵ
        x2=(x1+x0)/2;
        if g(x0)*g(x2) < 0 
            x1=x2
        else
            x0=x2
        end
    end
    σ3σ3min = x2
    x0=σ3mle
    x1=σ3max
    x2=(x1+x0)/2;
    while abs(x1-x0) > ϵ
        x2=(x1+x0)/2;
        if g(x0)*g(x2) < 0 
            x1=x2
        else
            x0=x2
        end
    end
    σ3σ3max = x2
    return σ3σ3min,σ3σ3max
end
σ3σ3min,σ3σ3max = Findinterceptσ3()
println("σ3confidence: ", [σ3σ3min,σ3σ3max])

# Plot the univariate profile likelihood for σ2 with σ1.
df = CSV.read("$path\\lls3.csv", DataFrames.DataFrame)
llσ3 = df[:, 1]
nllσ3=llσ3.-maximum(llσ3)
s3=plot!(σ3range,nllσ3,lw=6,xlim=(0.05,0.11),ylim=(-3,0.1),legend=false,tickfontsize=20,labelfontsize=20,margin = 10mm,grid=false,
xticks=([0.05,0.08,0.11], [latexstring("0.05"),latexstring("0.08"),latexstring("0.11")]),yticks=([-3,-2,-1,0],[latexstring("-3"),latexstring("-2"),latexstring("-1"),latexstring("0")]),framestyle=:box,color=1, line=:dot,left_margin = 15mm)
s3=vline!([σ3mle],lw=6,color=3, line=:dot)
s3=hline!([llstar],lw=6,color=2)
s3 = annotate!(0.074, -3.6, text(L"σ_{1}, σ_{2}, σ_{3} ", :left, 20))
s3 = annotate!(0.0365, -1.5, text(L"\bar{\ell}_{p}", :left, 20))
savefig(s3,"$path\\FigureS7c.pdf")  
display(s3) 

println("Profile σ3 Complete-----------------------------------------------------------------------------")


# Sample the parameters from rejection sampling.
# -------------------------------------------------
# -------------------------------------------------
# Notice: If the parameters have already been sampled and stored in lsp.csv, the prediction interval can be constructed directly using lsp.csv.
# -------------------------------------------------
# -------------------------------------------------
# Sample the parameters.
D1_iv = [0.15,0.38]
D2_iv = [0.13,0.28]
D3_iv = [0.01,0.45]
σ1_iv = [0.05,0.085]
σ2_iv = [0.05,0.08]
σ3_iv = [0.07,0.11]
M = 500
@time lsp =  sample_para(D1_iv,D2_iv,D3_iv,σ1_iv,σ2_iv,σ3_iv,fmle,M)
# Store sampled parameter set in lsp.csv
df = DataFrame(Value = lsp)
CSV.write("$path\\lsp.csv", df)

# Plot the parameters and their lower and upper bounds for rejection sampling.
D1sampled = [Para[1] for Para in lsp]
D2sampled = [Para[2] for Para in lsp]
D3sampled = [Para[3] for Para in lsp]
σ1sampled = [Para[4] for Para in lsp]
σ2sampled = [Para[5] for Para in lsp]
σ3sampled = [Para[6] for Para in lsp]

q1=scatter(D1sampled,legend=false,grid=false,xticks = false,tickfontsize = 15,size=(600, 200),markersize = 3,
ylims = (D1_iv[1]-0.01,D1_iv[2]+0.01),yticks=(D1_iv, [ string(D1_iv[1]), string(D1_iv[2])]),markerstrokecolor = :white, color=:black)
q1=hline!(D1_iv,legend=false,lw = 4, linestyle = :dash, color=:blue)
savefig(q1,"$path\\inval_D1.pdf")
display(q1)

q2=scatter(D2sampled,legend=false,grid=false,xticks = false,tickfontsize = 15,size=(600, 200),markersize = 3,
ylims = (D2_iv[1]-0.01,D2_iv[2]+0.01),yticks=(D2_iv, [ string(D2_iv[1]), string(D2_iv[2])]),markerstrokecolor = :white, color=:black)
q2=hline!(D2_iv,legend=false,lw = 4, linestyle = :dash, color=:blue)
savefig(q2,"$path\\inval_D2.pdf")
display(q2)

q3=scatter(D3sampled,legend=false,grid=false,xticks = false,tickfontsize = 15,size=(600, 200),markersize = 3,
ylims = (D3_iv[1]-0.01,D3_iv[2]+0.01),yticks=(D3_iv, [string(D3_iv[1]), string(D3_iv[2])]),markerstrokecolor = :white, color=:black)
q3=hline!(D3_iv,legend=false,lw = 4, linestyle = :dash, color=:blue)
savefig(q3,"$path\\inval_D3.pdf")
display(q3)

q4=scatter(σ1sampled,legend=false,grid=false,xticks = false,tickfontsize = 15,size=(600, 200),markersize = 3,
ylims = (σ1_iv[1]-0.01,σ1_iv[2]+0.01),yticks=(σ1_iv, [string(σ1_iv[1]), string(σ1_iv[2])]),markerstrokecolor = :white, color=:black)
q4=hline!(σ1_iv,legend=false,lw = 4, linestyle = :dash, color=:blue)
savefig(q4,"$path\\inval_sig1.pdf")
display(q4)

q5=scatter(σ2sampled,legend=false,grid=false,xticks = false,tickfontsize = 15,size=(600, 200),markersize = 3,
ylims = (σ2_iv[1]-0.01,σ2_iv[2]+0.01),yticks=(σ2_iv, [string(σ2_iv[1]), string(σ2_iv[2])]),markerstrokecolor = :white, color=:black)
q5=hline!(σ2_iv,legend=false,lw = 4, linestyle = :dash, color=:blue)
savefig(q5,"$path\\inval_sig2.pdf")
display(q5)

q6=scatter(σ3sampled,legend=false,grid=false,xticks = false,tickfontsize = 15,size=(600, 200),markersize = 3,
ylims = (σ3_iv[1]-0.001,σ3_iv[2]+0.001),yticks=(σ3_iv, [string(σ3_iv[1]), string(σ3_iv[2])]),markerstrokecolor = :white, color=:black)
q6=hline!(σ3_iv,legend=false,lw = 4, linestyle = :dash, color=:blue)
savefig(q6,"$path\\inval_sig3.pdf")
display(q6)

println("Parametes Select Complete---------------------------------------------------------------") 

# The data realisations prediction.

# Read the sampled parameter set from lsp.csv if saved
df = CSV.read("$path\\lsp.csv", DataFrames.DataFrame)
lsp1 = df[:, 1]

# Function to convert string tuple to numerical tuple
function parse_tuple(t::String)
    # Parse the string into Julia code
    parsed = Meta.parse(t)

    # Extract and convert to a tuple of numbers
    # Use `Tuple` to ensure it's a tuple and `Float64` for numerical conversion
    return Tuple(Float64.(eval(parsed)))
end
# Sampled parameter set.
lsp = [parse_tuple(x) for x in lsp1]

# Construct prediction interval.
t = 2000
@time lb1,ub1,lb2,ub2,lb3,ub3=predicreal(lsp,t)
lb = Float64.(lb); lb1 = Float64.(lb1); lb3 = Float64.(lb3);
ub = Float64.(ub); ub1 = Float64.(ub1); ub3 = Float64.(ub3);
x = 1:δ:L+1

f1 = plot(x, lb1.*J, lw=0, fillrange=ub1.*J, fillalpha=0.40, xlims = (0,200),ylims =(-5,25),color=:red, label=false, grid=false,tickfontsize = 20,margin = 10mm,framestyle=:box,
xticks=([1,50,100,150,200], [latexstring("1"),"",latexstring("100"),"",latexstring("200")]),yticks=([0,5,10,15,20], [latexstring("0"),"","","",latexstring("20")]),linecolor=:transparent)
f1 = plot!(x,C1mle.*J,color=:red,legend=false,lw=4,ls=:dash)
f1 = scatter!(1:1:200,data1, markersize = 4, markerstrokecolor=:blue, color=:blue)
f1 = hline!([0,20],legend=false,lw = 4, linestyle = :dash, color=:black)
f1 = annotate!(100, -11, text(L"i ", :left, 20))
f1 = annotate!(-200*0.17, 10, text(L"C_{1}^{\mathrm{o}} ", :left, 20))
savefig(f1,"$path\\FigureS8a.pdf")
display(f1)

f2 = plot(x, lb2.*J, lw=0, fillrange=ub2.*J, fillalpha=0.40, xlims = (0,200),ylims =(-5,25),color=:blue, label=false, grid=false,tickfontsize = 20,margin = 10mm,framestyle=:box,
xticks=([1,50,100,150,200], [latexstring("1"),"",latexstring("100"),"",latexstring("200")]),yticks=([0,5,10,15,20], [latexstring("0"),"","","",latexstring("20")]),linecolor=:transparent)
f2 = plot!(x,C2mle.*J,color=:blue,legend=false,lw=4,ls=:dash)
f2 = scatter!(1:1:200,data2, markersize = 4, markerstrokecolor=:blue, color=:blue)
f2 = hline!([0,20],legend=false,lw = 4, linestyle = :dash, color=:black)
f2 = annotate!(100, -11, text(L"i ", :left, 20))
f2 = annotate!(-200*0.17, 10, text(L"C_{2}^{\mathrm{o}} ", :left, 20))
savefig(f2,"$path\\FigureS8c.pdf")
display(f2)

f3 = plot(x, lb3.*J, lw=0, fillrange=ub3.*J, fillalpha=0.40, xlims = (0,200),ylims =(-5,25),color=:green, label=false, grid=false,tickfontsize = 20,margin = 10mm,framestyle=:box,
xticks=([1,50,100,150,200], [latexstring("1"),"",latexstring("100"),"",latexstring("200")]),yticks=([0,5,10,15,20], [latexstring("0"),"","","",latexstring("20")]),linecolor=:transparent)
f3 = plot!(x,C3mle.*J,color=:green,legend=false,lw=4,ls=:dash)
f3 = scatter!(1:1:200,data3, markersize = 4, markerstrokecolor=:blue, color=:blue)
f3 = hline!([0,20],legend=false,lw = 4, linestyle = :dash, color=:black)
f3 = annotate!(100, -11, text(L"i ", :left, 20))
f3 = annotate!(-200*0.17, 10, text(L"C_{3}^{\mathrm{o}} ", :left, 20))
savefig(f3,"$path\\FigureS8e.pdf")
display(f3)

println("realisation Prediction Complete---------------------------------------------------------------")

# Prediction interval coverage.
flb1 = linear_interpolation(x,vec(lb1.*J));
fub1 = linear_interpolation(x,vec(ub1.*J));
flb2 = linear_interpolation(x,vec(lb2.*J));
fub2 = linear_interpolation(x,vec(ub2.*J));
flb3 = linear_interpolation(x,vec(lb3.*J));
fub3 = linear_interpolation(x,vec(ub3.*J));
global o1 = 0
global o2 = 0
global o3 = 0
for i = 1:1:200
    if data1[i] > fub1(i) || data1[i] < flb1(i)
        global o1 = o1 + 1
    end

    if data2[i] > fub2(i) || data2[i] < flb2(i)
        global o2 = o2 + 1
    end

    if data3[i] > fub3(i) || data3[i] < flb3(i)
        global o3 = o3 + 1
    end
end

out_rate = (o1+o2+o3)/(length(data1)+length(data2)+length(data3))* 100
println("The number of data outside interval for subpopulation 1: ",o1)
println("The number of data outside interval for subpopulation 2: ",o2)
println("The number of data outside interval for subpopulation 3: ",o3)

println("The percentage of data outside interval: $out_rate %")