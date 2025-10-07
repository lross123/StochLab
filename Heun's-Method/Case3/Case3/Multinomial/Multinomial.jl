using Plots, DifferentialEquations
using CSV
using DataFrames
using LaTeXStrings
using Interpolations
using Distributions
using Measures
using NLopt
using FilePathsBase

# In this script, we generate the result for multinomial measurement error model for Case 3

# Get the path of the current script
path = dirname(@__FILE__)

# Read count data from the CSV file.
data = DataFrame(CSV.File("$path\\data.csv"));
# Count data for subpopulation 1, 2 and 3 in Case 3.
# Here, data1 corresponds to C_1^{o}, data2 corresponds to C_2^{o} and data2 corresponds to C_3^{o}.
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


# model(): The continuum model used for Case 2.
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


# error(): Log-likelihood function for the multinomial Measurement Error Model for Case 3.
function error(data1,data2,data3,δ,lx,a) 
    #------------------------------------------------------------------------------------------------------------------
    # Input:
    # data1: (vector) Count data for subpopulation 1.
    # data2: (vector) Count data for subpopulation 2.
    # data3: (vector) Count data for subpopulation 3.
    # δ: (number) Grid spacing.
    # L: (number) PDE solved on 0 <= x <= L.
    # a: (vector) Parameter vector.
    # Output:
    # e: (number) Log-likelihood for the multinomial Measurement Error Model (Case 3) with parameter vector a.
    #------------------------------------------------------------------------------------------------------------------
    # Solve the PDE at time t1 with parameter vector a.
    c1,c2,c3=model(t1,a,δ,L);
    # Find c1(x_i, t1) and c2(x_i, t1) for i = 1, 2, 3, ..., 200.
    xlocdata = 0:δ:L
    fc1= linear_interpolation(xlocdata,c1);
    fc2= linear_interpolation(xlocdata,c2);
    fc3= linear_interpolation(xlocdata,c3);
    # Estimate the loglikelihood function.
    e=0.0;
    for i = 1:lx+1
        c1 = fc1(i-1); c2 = fc2(i-1); c3 = fc3(i-1) 
        e = e + log((c1^data1[i]) * (c2^data2[i])* (c3^data3[i]) * ((1-c1-c2-c3)^(J-data1[i]-data2[i]-data3[i])))
    end 
    return e
end

# fun(): Evaluate the log-likelihood function as a function of the parameters a.
function fun(a)
    # ------------------------------------------------------------------------------------------------------------------
    # Input:
    # a: (vector) Parameter vector.
    # Output:
    # error(data1, data2, data3, δ, lx, a): (number) Log-likelihood for the multinomial Measurement Error Model (Case 2) with parameter vector a.
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
function sample_para(D1_iv,D2_iv,D3_iv,fmle,M)
    # ------------------------------------------------------------------------------------------------------------------
    # Input:
    # D1_iv, D2_iv, D3_iv: (vector) Intervals for D1, D2 and D3 for rejection sampling.
    # fmle: (number) Log-likelihood at MLE.
    # M: (number) Number of parameter sets required within the 95% log-likelihood threshold.
    # Output:
    # sampledP: (vector) M parameter sets within the 95% log-likelihood threshold.
    # ------------------------------------------------------------------------------------------------------------------
    # Degree of freedom.
    df=3
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
        D2g=rand(Uniform(D2_iv[1],D2_iv[2]))
        D3g=rand(Uniform(D3_iv[1],D3_iv[2]))
        # Evaluate the log-likelihood function using sampled parameter sets.
        ll = fun([D1g,D2g,D3g])
        if ll-fmle >= llstar
            # Sampled parameter set within the 95% log-likelihood threshold.
            # Update number of sampled parameter sets within the 95% log-likelihood threshold.
            count = count + 1
            # Update sampled parameter sets within the 95% log-likelihood threshold.
            push!(sampledP,(D1g,D2g,D3g))
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
    # ------------------------------------------------------------------------------------------------------------------
    # Generate empty lower and upper bounds of prediction intervals for subpopulations 1 and 2.
    xx = 0:δ:L
    gp=length(xx)
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
            ulb1 = (quantile(Binomial(J,p1(xx[j])),[.05,.95])[1])/J
            uub1 = (quantile(Binomial(J,p1(xx[j])),[.05,.95])[2])/J
            ulb2 = (quantile(Binomial(J,p2(xx[j])),[.05,.95])[1])/J
            uub2 = (quantile(Binomial(J,p2(xx[j])),[.05,.95])[2])/J
            ulb3 = (quantile(Binomial(J,p3(xx[j])),[.05,.95])[1])/J
            uub3 = (quantile(Binomial(J,p3(xx[j])),[.05,.95])[2])/J

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
D1g = 0.18; D2g = 0.3; D3g = 0.2
θG = [D1g,D2g,D3g] # inital guess
lb=[0.01,0.01,0.01] #lower bound
ub=[0.4,0.4,0.4] #upper bound
# Call numerical optimization routine to give the vector of parameters xopt, and the maximum loglikelihood fopt.
@time (xopt,fopt)  = optimise(fun,θG,lb,ub)
fmle=fopt
# Print MLE parameters
D1mle=xopt[1]; 
D2mle=xopt[2]; 
D3mle=xopt[3];
println("D1mle: ", D1mle)
println("D2mle: ", D2mle)
println("D3mle: ", D3mle)
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
D1min=0.2 # lower bound for the profile 
D1max=0.35 # upper bound for the profile 
D1range=LinRange(D1min,D1max,nptss) # vector of D1 values along the profile
nrange=zeros(2,nptss) # matrix to store the nuisance parameters once optimized out
llD1=zeros(nptss) # loglikelihood at each point along the profile
nllD1=zeros(nptss) # normalised loglikelihood at each point along the profile

@time for i in 1:nptss
    function fun1(aa) # function to return loglikelihood by fixing the interest parameter along the profile
        return error(data1,data2,data3,δ,L,[D1range[i],aa[1],aa[2]])
    end
    local lb1=[0.01,0.01] # lower bound 
    local ub1=[0.4,0.4] # upper bound
    local θG1=[D2mle,D3mle] # initial estimate - take the MLE 
    local (xo,fo)=optimise(fun1,θG1,lb1,ub1)
    nrange[:,i]=xo[:] # store the nuisance parameters
    llD1[i]=fo[1] # store the loglikelihood 
end
nllD1=llD1.-maximum(llD1); # calculate normalised loglikelihood
# Store the loglikelihood in csv.
df = DataFrame(Value = llD1)
CSV.write("$path\\llD1.csv", df)
# Plot the univariate profile likelihood for D1 and hold.
df = CSV.read("$path\\llD1.csv", DataFrames.DataFrame)
llD1 = df[:, 1]
nllD1=llD1.-maximum(llD1)
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
D2min=0.15 # lower bound for the profile 
D2max=0.25 # upper bound for the profile 
D2range=LinRange(D2min,D2max,nptss) # vector of D2 values along the profile
nrange=zeros(2,nptss) # matrix to store the nuisance parameters once optimized out
llD2=zeros(nptss) # loglikelihood at each point along the profile
nllD2=zeros(nptss) # normalised loglikelihood at each point along the profile

@time for i in 1:nptss
    function fun2(aa) # function to return loglikelihood by fixing the interest parameter along the profile
        return error(data1,data2,data3,δ,L,[aa[1],D2range[i],aa[2]])
    end
    local lb2=[0.01,0.01] # lower bound 
    local ub2=[0.4,0.4] # upper bound
    local θG2=[D1mle,D3mle] # initial estimate - take the MLE 
    local (xo,fo)=optimise(fun2,θG2,lb2,ub2)
    nrange[:,i]=xo[:] # store the nuisance parameters
    llD2[i]=fo[1] # store the loglikelihood 
end
nllD2=llD2.-maximum(llD2); # calculate normalised loglikelihood
# Store the loglikelihood in csv.
df = DataFrame(Value = llD2)
CSV.write("$path\\llD2.csv", df)

# Plot the univariate profile likelihood for D2 with D1 and hold.
df = CSV.read("$path\\llD2.csv", DataFrames.DataFrame)
llD2 = df[:, 1]
nllD2=llD2.-maximum(llD2)
s1=plot!(D2range,nllD2,lw=6,xlim=(0,0.4),ylim=(-3,0.1),legend=false,tickfontsize=20,labelfontsize=20,margin = 10mm,grid=false,
xticks=([0,0.1,0.2,0.3,0.4], [latexstring("0"),"", latexstring("0.2"),"",latexstring("0.4")]),yticks=([-3,-2,-1,0],[latexstring("-3"),latexstring("-2"),latexstring("-1"),latexstring("0")]),framestyle=:box,color=1,line=:dashdot,left_margin = 15mm)
s1=vline!([D2mle],lw=6,color=3,line=:dashdot)
s1=hline!([llstar],lw=6,color=2)

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


# Profile D3-------------------------------------------------------------------
df = 1
llstar = -quantile(Chisq(df),0.95)/2 # 95% asymptotic threshold
nptss=40 # points on the profile
D3min=0.01 # lower bound for the profile 
D3max=0.2 # upper bound for the profile 
D3range=LinRange(D3min,D3max,nptss) # vector of D3 values along the profile
nrange=zeros(2,nptss) # matrix to store the nuisance parameters once optimized out
llD3=zeros(nptss) # loglikelihood at each point along the profile
nllD3=zeros(nptss) # normalised loglikelihood at each point along the profile

@time for i in 1:nptss
    function fun3(aa) # function to return loglikelihood by fixing the interest parameter along the profile
        return error(data1,data2,data3,δ,L,[aa[1],aa[2],D3range[i]])
    end
    local lb3=[0.01,0.01] # lower bound 
    local ub3=[0.4,0.4] # upper bound
    local θG3=[D1mle,D2mle] # initial estimate - take the MLE 
    local (xo,fo)=optimise(fun3,θG3,lb3,ub3)
    nrange[:,i]=xo[:] # store the nuisance parameters
    llD3[i]=fo[1] # store the loglikelihood 
end
nllD3=llD3.-maximum(llD3); # calculate normalised loglikelihood
# Store the loglikelihood in csv.
df = DataFrame(Value = llD3)
CSV.write("$path\\llD3.csv", df)

# Plot the univariate profile likelihood for D3 with D1 and D2.
df = CSV.read("$path\\llD3.csv", DataFrames.DataFrame)
llD3 = df[:, 1]
nllD3=llD3.-maximum(llD3)
s1=plot!(D3range,nllD3,lw=6,xlim=(0,0.4),ylim=(-3,0.1),legend=false,tickfontsize=20,labelfontsize=20,margin = 10mm,grid=false,
xticks=([0,0.1,0.2,0.3,0.4], [latexstring("0"),"", latexstring("0.2"),"",latexstring("0.4")]),yticks=([-3,-2,-1,0],[latexstring("-3"),latexstring("-2"),latexstring("-1"),latexstring("0")]),framestyle=:box,color=1,line=:dot,left_margin = 15mm)
s1=vline!([D3mle],lw=6,color=3,line=:dot)
s1=hline!([llstar],lw=6,color=2)
s1 = annotate!(0.145, -3.6, text(L"D_{1}, D_{2}, D_{3} ", :left, 20))
s1 = annotate!(-0.09, -1.5, text(L"\bar{\ell}_{p}", :left, 20))
savefig(s1,"$path\\FigureS7b.pdf")
display(s1) 

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
println("Profile D3 Complete-----------------------------------------------------------------------------")

# Sample the parameters from rejection sampling.
# -------------------------------------------------
# -------------------------------------------------
# Notice: If the parameters have already been sampled and stored in lsp.csv, the prediction interval can be constructed directly using lsp.csv.
# -------------------------------------------------
# -------------------------------------------------
# sample the parameters
D1_iv = [0.15,0.35]
D2_iv = [0.1,0.3]
D3_iv = [0.01,0.3]
M = 500
@time lsp =  sample_para(D1_iv,D2_iv,D3_iv,fmle,M)
# Store sampled parameter set in lsp.csv
df = DataFrame(Value = lsp)
CSV.write("$path\\lsp.csv", df)

# Plot the parameters and their lower and upper bounds for rejection sampling.
D1sampled = [Para[1] for Para in lsp]
D2sampled = [Para[2] for Para in lsp]
D3sampled = [Para[3] for Para in lsp]
q1=scatter(D1sampled,legend=false,grid=false,xticks=([1,500],["1","500"]),tickfontsize = 15,size=(600, 200),markersize = 3,
ylims = (D1_iv[1]-0.1,D1_iv[2]+0.1),yticks=(D1_iv , [ string(D1_iv[1]), string(D1_iv[2])]),markerstrokecolor = :white, color=:black)
q1=hline!(D1_iv,legend=false,lw = 4, linestyle = :dash, color=:blue)
savefig(q1,"$path\\inval_D1.pdf")
display(q1)

q2=scatter(D2sampled,legend=false,grid=false,xticks=([1,500],["1","500"]),tickfontsize = 15,size=(600, 200),markersize = 3,
ylims = (D2_iv[1]-0.1,D2_iv[2]+0.1),yticks=(D2_iv, [ string(D2_iv[1]), string(D2_iv[2])]),markerstrokecolor = :white, color=:black)
q2=hline!(D2_iv,legend=false,lw = 4, linestyle = :dash, color=:blue)
savefig(q2,"$path\\inval_D2.pdf")
display(q2)

q3=scatter(D3sampled,legend=false,grid=false,xticks=([1,500],["1","500"]),tickfontsize = 15,size=(600, 200),markersize = 3,
ylims = (D3_iv[1]-0.1,D3_iv[2]+0.1),yticks=(D3_iv, [ string(D3_iv[1]), string(D3_iv[2])]),markerstrokecolor = :white, color=:black)
q3=hline!(D3_iv,legend=false,lw = 4, linestyle = :dash, color=:blue)
savefig(q3,"$path\\inval_D3.pdf")
display(q3)



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
savefig(f1,"$path\\FigureS8b.pdf")
display(f1)

f2 = plot(x, lb2.*J, lw=0, fillrange=ub2.*J, fillalpha=0.40, xlims = (0,200),ylims =(-5,25),color=:blue, label=false, grid=false,tickfontsize = 20,margin = 10mm,framestyle=:box,
xticks=([1,50,100,150,200], [latexstring("1"),"",latexstring("100"),"",latexstring("200")]),yticks=([0,5,10,15,20], [latexstring("0"),"","","",latexstring("20")]),linecolor=:transparent)
f2 = plot!(x,C2mle.*J,color=:blue,legend=false,lw=4,ls=:dash)
f2 = scatter!(1:1:200,data2, markersize = 4, markerstrokecolor=:blue, color=:blue)
f2 = hline!([0,20],legend=false,lw = 4, linestyle = :dash, color=:black)
f2 = annotate!(100, -11, text(L"i ", :left, 20))
f2 = annotate!(-200*0.17, 10, text(L"C_{2}^{\mathrm{o}} ", :left, 20))
savefig(f2,"$path\\FigureS8d.pdf")
display(f2)

f3 = plot(x, lb3.*J, lw=0, fillrange=ub3.*J, fillalpha=0.40, xlims = (0,200),ylims =(-5,25),color=:green, label=false, grid=false,tickfontsize = 20,margin = 10mm,framestyle=:box,
xticks=([1,50,100,150,200], [latexstring("1"),"",latexstring("100"),"",latexstring("200")]),yticks=([0,5,10,15,20], [latexstring("0"),"","","",latexstring("20")]),linecolor=:transparent)
f3 = plot!(x,C3mle.*J,color=:green,legend=false,lw=4,ls=:dash)
f3 = scatter!(1:1:200,data3, markersize = 4, markerstrokecolor=:blue, color=:blue)
f3 = hline!([0,20],legend=false,lw = 4, linestyle = :dash, color=:black)
f3 = annotate!(100, -11, text(L"i ", :left, 20))
f3 = annotate!(-200*0.17, 10, text(L"C_{3}^{\mathrm{o}} ", :left, 20))
savefig(f3,"$path\\FigureS8f.pdf")
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
println("The number of data outside interval for subpopulation 2: ",o3)

println("The percentage of data outside interval: $out_rate %")