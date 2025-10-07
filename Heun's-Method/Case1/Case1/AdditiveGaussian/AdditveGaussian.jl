using Plots, DifferentialEquations
using CSV
using DataFrames
using LaTeXStrings
using Interpolations
using Distributions
using Measures
using NLopt
using FilePathsBase
# In this script, we generate the result for additive Gaussian measurement error model when there is only one subpopulation involved.

# Get the path of the current script.
path = dirname(@__FILE__)

# Read count data from the CSV file.
data = DataFrame(CSV.File("$path\\data.csv")); 
# Count data for subpopulation 1 in Case 1.
# Here, data1 corresponds to C_1^{o}
data1 = data.a
# Fixed parameters
# Data is collected at time t = t1.
t1 = 300
# The PDE is solved on 0<=x<=L with grid spacing δ.
δ=0.5; L=199; J=20

# diff!(): Is the discretized PDE allowing the equation to be solved via numerical methods.
function diff!(d_u, u, p, t)
    # ------------------------------------------------------------------------------------------------------------------
    # Input:
    # d_u: (vector) d_u is dc1/dt.
    # u: (vector)  u[n] is c1(x_n)
    # p: (vector) [D1,v1,δ].
    # t: (number) Time.
    # Output:
    # d_u: (matrix) d_u is dc1/dt.
    # ------------------------------------------------------------------------------------------------------------------
    (D1,v1,δ) = p   
    N = length(u)

    # The equation for the boundary at x = 0.
    d_u[1] = (D1/(δ^2))*(u[2]-u[1]) -  (v1/δ)*(u[2]*(1-u[2]))

    # The equations for between x=0 and x=199.
    for n in 2:N-1
        d_u[n] = (D1/(δ^2))*(u[n+1]-2*u[n]+u[n-1]) - (v1/(2*δ))*(u[n+1]*(1-u[n+1]) - u[n-1]*(1-u[n-1]))
    end 

    # The equation for the boundary at x = 199.
    d_u[N] = (-D1/(δ^2))*(u[N] - u[N-1]) + (v1/δ)*(u[N-1]*(1-u[N-1]))

    return d_u
end

# pdesolver(): Solves the ODE in diff!().
function pdesolver(Time,δ,L,D1,v1)
    # ------------------------------------------------------------------------------------------------------------------
    # Input:
    # Time: (number) Solve ODE/PDE at t = Time.
    # δ: (number) Grid spacing.
    # L: (number) ODE/PDE is solved on 0 <= x <= L.
    # D1: (number) Diffusivity for agents in subpopulation 1.
    # v1: (number) Drift velocity for agents in subpopulation 1.
    # Output:
    # c1: (vector) Solution of ODE/PDE at time t = Time.
    # ------------------------------------------------------------------------------------------------------------------
    # Construct initial condition for PDE.
    # Total number of mesh points N.
    N = Int(L/δ + 1)
    # Initial condition u0.
    u0 = zeros(N,1)
    # Update initial condition u0: u0=1 at 9<=x<=39.
    for n=1:N
        xx = (n-1)*δ
        if xx >=9 && xx <= 39
           u0[n] = 1
        end
    end
    
    # Return the inital conditon if ODE/PDE is solved at t=0.
    if Time == 0
        return u0
    end

    # Solve the PDE using Heun's method at t=Time.
    p=(D1,v1,δ)
    tspan=(0,Time)
    alg=Heun() 
    prob=ODEProblem(diff!,u0,tspan,p)
    sol=solve(prob,alg,saveat=Time);
    c1 = sol[end]
    return c1
end 

# model(): The continuum model used for Case 1.
function model(t1,δ,L,a) 
    #------------------------------------------------------------------------------------------------------------------
    # Input:
    # t1: (number) Solve the PDE at t = t1.
    # δ: (number) Grid spacing.
    # L: (number) PDE is solved on 0 <= x <= L.
    # a: (vector) Parameter vector.
    # Output:
    # c1: (vector) Solution of the PDE at time t = Time.
    #------------------------------------------------------------------------------------------------------------------
    # Solve the PDE.
    c1=pdesolver(t1,δ,L,a[1],a[2]) 
    return c1
end

# error(): Log-likelihood function for the Additive Gaussian Measurement Error Model for Case 1.
function error(data1,t1,δ,L,a)
    #------------------------------------------------------------------------------------------------------------------
    # Input:
    # data1: (vector) Count data for subpopulation 1.
    # δ: (number) Grid spacing.
    # L: (number) PDE solved on 0 <= x <= L.
    # a: (vector) Parameter vector.
    # Output:
    # e: (number) Log-likelihood for the Additive Gaussian Measurement Error Model (Case 1) with parameter vector a.
    #------------------------------------------------------------------------------------------------------------------
    # Solve the PDE at time t1 with parameter vector a.
    c1=model(t1,δ,L,a)
    # Find c1(x_i, t1) for i = 1, 2, 3, ..., 200.
    xlocdata = 0:δ:L
    interpr = linear_interpolation(xlocdata,vec(c1));
    c1 = interpr(0:1:L)
    # Estimate the loglikelihood function.
    e=0.0;
    dist1 = Normal(0,a[3])
    e=loglikelihood(dist1,(data1./J).-c1)
    return e
end

# fun(): Evaluate the log-likelihood function as a function of the parameters a.
function fun(a)
    # ------------------------------------------------------------------------------------------------------------------
    # Input:
    # a: (vector) [D1,v1].
    # Output:
    # error(data1,t1,δ,lx,a): (number) Log-likelihood for the Additive Gaussian Measurement Error Model (Case 1) with parameter vector a.
    #------------------------------------------------------------------------------------------------------------------
    return error(data1,t1,δ,L,a)
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
    opt.maxtime = 10*60
    res = optimize(opt,θ₀)
    return res[[2,1]]
end

# sample_para(): Perform rejection sampling to select M parameter sets within the 95% log-likelihood threshold.
function sample_para(D1_iv,v1_iv,σ1_iv,fmle,M)
    # ------------------------------------------------------------------------------------------------------------------
    # Input:
    # D1_iv, v1_iv, σ1_iv: (vector) Intervals for D1, v1 and σ1 for rejection sampling.
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
        v1g=rand(Uniform(v1_iv[1],v1_iv[2]))
        σ1g=rand(Uniform(σ1_iv[1],σ1_iv[2]))
        # Evaluate the log-likelihood function using sampled parameter sets.
        ll = fun([D1g,v1g,σ1g])
        if ll-fmle >= llstar
            # Sampled parameter set within the 95% log-likelihood threshold.
            # Update number of sampled parameter sets within the 95% log-likelihood threshold.
            count = count + 1
            # Update sampled parameter sets within the 95% log-likelihood threshold.
            push!(sampledP,(D1g,v1g,σ1g))
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
    # ------------------------------------------------------------------------------------------------------------------
    # Generate empty lower and upper bounds of prediction intervals for subpopulations 1.
    xx = 0:δ:L
    gp=length(xx)
    lb1 = ones(gp)
    ub1 = zeros(gp)

    for i = 1:length(lsp)
        # Solve the PDE with parameter set a.
        a = lsp[i]
        c1 = model(t,δ,L,[a[1],a[2]])
        p = linear_interpolation(xx,vec(c1));

        # Construct prediction interval for data realizations.
        for j = 1:length(xx)
            c1_05 = (quantile(Normal(p(xx[j]),a[3]),[.05,.95])[1])
            c1_95 = (quantile(Normal(p(xx[j]),a[3]),[.05,.95])[2])
            if c1_05 < lb1[j] 
                lb1[j] = c1_05
            end

            if c1_95 > ub1[j] 
                ub1[j] = c1_95
            end
        end
    end
    return lb1,ub1
end

#Expected Parameters
P1 = 1; ρ1 = 0.1
D1 = P1/4; v1 = (ρ1*P1)/(2)

#MLE----------------------------------------------------------------------------
# Inital guess.
D1g = 0.1; v1g = 0.01; σ1g = 0.1;
θG = [D1g,v1g,σ1g] # inital guess
lb=[0.01,0,0] # lower bound
ub=[0.3,0.1,1] # upper bound
# Call numerical optimization routine to give the vector of parameters xopt, and the maximum loglikelihood fopt.
@time (xopt,fopt)  = optimise(fun,θG,lb,ub) 
fmle=fopt
# Print MLE parameters.
D1mle=xopt[1]; 
v1mle=xopt[2]; 
σ1mle=xopt[3]; 
println("D1mle: ", D1mle)
println("v1mle: ", v1mle)
println("σ1mle: ", σ1mle)
# Solve the PDE with MLE parameters
Cmle = model(t1,δ,L,[D1mle,v1mle])  
println("MLE Complete---------------------------------------------------------------")

# Profile D1-------------------------------------------------------------------
df = 1
llstar = -quantile(Chisq(df),0.95)/2 # 95% asymptotic threshold
nptss=40 # points on the profile
D1min=0.2 # lower bound for the profile 
D1max=0.3 # upper bound for the profile 
D1range=LinRange(D1min,D1max,nptss) # vector of D1 values along the profile
nrange=zeros(2,nptss) # matrix to store the nuisance parameters once optimized out
llD1=zeros(nptss) # loglikelihood at each point along the profile
nllD1=zeros(nptss) # normalised loglikelihood at each point along the profile

@time for i in 1:nptss
    function fun1(aa) # function to return loglikelihood by fixing the interest parameter along the profile
        return error(data1,t1,δ,L,[D1range[i],aa[1],aa[2]])
    end
    local lb1=[0,0] # lower bound 
    local ub1=[0.1,1] # upper bound
    local θG1=[v1mle,σ1mle] # initial estimate - take the MLE 
    local (xo,fo)=optimise(fun1,θG1,lb1,ub1)
    nrange[:,i]=xo[:] # store the nuisance parameters
    llD1[i]=fo[1] # store the loglikelihood 
end
nllD1=llD1.-maximum(llD1); # calculate normalised loglikelihood
println("Profile D1 Complete-----------------------------------------------------------------------------")

# Plot the univariate profile likelihood for D1.
s1=plot(D1range,nllD1,lw=6,xlim=(0,0.4),ylim=(-3,0.1),legend=false,tickfontsize=20,labelfontsize=20,margin = 10mm,grid=false,
xticks=([0,0.1,0.2,0.3,0.4], [latexstring("0"),"", latexstring("0.2"),"",latexstring("0.4")]),yticks=([-3,-2,-1,0],[latexstring("-3"),latexstring("-2"),latexstring("-1"),latexstring("0")]),framestyle=:box,left_margin = 15mm)
s1=hline!([llstar],lw=6)
s1=vline!([D1mle],lw=6)
s1 = annotate!(0.2, -3.6, text(L"D_{1} ", :left, 20))
s1 = annotate!(-0.09, -1.5, text(L"\bar{\ell}_{p}", :left, 20))
savefig(s1,"$path\\Figure3a.pdf")
display(s1)

# Profile v1-------------------------------------------------------------------
df = 1
llstar = -quantile(Chisq(df),0.95)/2 # 95% asymptotic threshold
nptss=40 #points on the profile
v1min=0.02#lower bound for the profile  
v1max=0.08#upper bound for the profile  
v1range=LinRange(v1min,v1max,nptss) #vector of v1 values along the profile
nrange=zeros(2,nptss) # matrix to store the nuisance parameters once optimized out
llv1=zeros(nptss) # loglikelihood at each point along the profile
nllv1=zeros(nptss) # normalised loglikelihood at each point along the profile

@time for i in 1:nptss
    function fun2(aa) # function to return loglikelihood by fixing the interest parameter along the profile
        return error(data1,t1,δ,L,[aa[1],v1range[i],aa[2]])
    end
    local lb2=[0.01,0] # lower bound 
    local ub2=[0.3,1] # upper bound
    local θG2=[D1mle,σ1mle] # initial estimate - take the MLE 
    local (xo,fo)=optimise(fun2,θG2,lb2,ub2)
    nrange[:,i]=xo[:] # store the nuisance parameters
    llv1[i]=fo[1] # store the loglikelihood 
end
nllv1=llv1.-maximum(llv1); # calculate normalised loglikelihood
println("Profile v1 Complete-----------------------------------------------------------------------------")

# Plot the univariate profile likelihood for v1.
s2=plot(v1range,nllv1,lw=6,xlim=(0,0.1),ylim=(-3,0.1),legend=false,tickfontsize=20,labelfontsize=20,margin = 10mm,grid=false,
xticks=([0,0.025,0.05,0.075,0.1], [latexstring("0"),"", latexstring("0.05"),"",latexstring("0.1")]),yticks=([-3,-2,-1,0],[latexstring("-3"),latexstring("-2"),latexstring("-1"),latexstring("0")]),framestyle=:box,left_margin = 15mm)
s2=hline!([llstar],lw=6)
s2=vline!([v1mle],lw=6)
s2 = annotate!(0.05, -3.6, text(L"v_{1} ", :left, 20))
s2 = annotate!(-0.09/4, -1.5, text(L"\bar{\ell}_{p}", :left, 20))
savefig(s2,"$path\\Figure3c.pdf")
display(s2)


#Profile σ1-------------------------------------------------------------------
df = 1
llstar = -quantile(Chisq(df),0.95)/2 # 95% asymptotic threshold
nptss=40 # points on the profile
σ1min=0.05 #lower bound for the profile 
σ1max=0.08 # upper bound for the profile 
σ1range=LinRange(σ1min,σ1max,nptss) # vector of σ1 values along the profile
nrange=zeros(2,nptss) # matrix to store the nuisance parameters once optimized out
llσ1=zeros(nptss) # loglikelihood at each point along the profile
nllσ1=zeros(nptss) # normalised loglikelihood at each point along the profile

@time for i in 1:nptss
    function fun3(aa) # function to return loglikelihood by fixing the interest parameter along the profile
        return error(data1,t1,δ,L,[aa[1],aa[2],σ1range[i]])
    end
    local lb3=[0.01,-1] # lower bound 
    local ub3=[0.3,1] # upper bound
    local θG3=[D1mle,v1mle] # initial estimate - take the MLE 
    local (xo,fo)=optimise(fun3,θG3,lb3,ub3)
    nrange[:,i]=xo[:] # store the nuisance parameters
    llσ1[i]=fo[1] # store the loglikelihood 
end
nllσ1=llσ1.-maximum(llσ1); # calculate normalised loglikelihood
println("Profile σ1 Complete-----------------------------------------------------------------------------")

# Plot the univariate profile likelihood for σ1.
s3=plot(σ1range,nllσ1,lw=6,xlim=(0,0.1),ylim=(-3,0.1),legend=false,tickfontsize=20,labelfontsize=20,margin = 10mm,grid=false,
xticks=([0,0.025,0.05,0.075,0.1], [latexstring("0"),"", latexstring("0.05"),"",latexstring("0.1")]),yticks=([-3,-2,-1,0],[latexstring("-3"),latexstring("-2"),latexstring("-1"),latexstring("0")]),framestyle=:box,left_margin = 15mm)
s3=hline!([llstar],lw=6)
s3=vline!([σ1mle],lw=6)
s3 = annotate!(0.05, -3.6, text(L"σ_{1} ", :left, 20))
s3 = annotate!(-0.09/4, -1.5, text(L"\bar{\ell}_{p}", :left, 20))
savefig(s3,"$path\\Figure3e.pdf")
display(s3)

# Sample the parameters from rejection sampling.
# -------------------------------------------------
# -------------------------------------------------
# Notice: If the parameters have already been sampled and stored in lsp.csv, the prediction interval can be constructed directly using lsp.csv.
# -------------------------------------------------
# -------------------------------------------------
# Sample the parameters.
D1_iv = [0.17,0.34]
v1_iv = [0.027,0.064]
σ1_iv = [0.038,0.074]
M = 500
@time lsp =  sample_para(D1_iv,v1_iv,σ1_iv,fmle,M)
# Store sampled parameter set in lsp.csv
df = DataFrame(Value = lsp)
CSV.write("$path\\lsp.csv", df)

# Plot the parameters and their lower and upper bounds for rejection sampling.
D1sampled = [Para[1] for Para in lsp]
v1sampled = [Para[2] for Para in lsp]
σ1sampled = [Para[3] for Para in lsp]

q1=scatter(D1sampled,legend=false,grid=false,xticks = false,tickfontsize = 15,size=(600, 200),markersize = 3,
ylims = (D1_iv[1]-0.05,D1_iv[2]+0.05),yticks=(D1_iv, [ string(D1_iv[1]), string(D1_iv[2])]),markerstrokecolor = :white, color=:black)
q1=hline!(D1_iv,legend=false,lw = 4, linestyle = :dash, color=:blue)
savefig(q1,"$path\\inval_D1.pdf")
display(q1)

q2=scatter(v1sampled,legend=false,grid=false,xticks = false,tickfontsize = 15,size=(600, 200),markersize = 3,
ylims = (v1_iv[1]-0.01,v1_iv[2]+0.01),yticks=(v1_iv, [ string(v1_iv[1]), string(v1_iv[2])]),markerstrokecolor = :white, color=:black)
q2=hline!(v1_iv,legend=false,lw = 4, linestyle = :dash, color=:blue)
savefig(q2,"$path\\inval_v1.pdf")
display(q2)

q3=scatter(σ1sampled,legend=false,grid=false,xticks = false,tickfontsize = 15,size=(600, 200),markersize = 3,
ylims = (σ1_iv[1]-0.005,σ1_iv[2]+0.005),yticks=(σ1_iv, [ string(σ1_iv[1]), string(σ1_iv[2])]),markerstrokecolor = :white, color=:black)
q3=hline!(σ1_iv,legend=false,lw = 4, linestyle = :dash, color=:blue)
savefig(q3,"$path\\inval_sig1.pdf")
display(q3)

println("Parametes Select Complete---------------------------------------------------------------")

# The data realisations prediction.

# Read the sampled parameter set from lsp.csv if saved
df = CSV.read("$path\\lsp.csv", DataFrames.DataFrame)
lsp = df[:, 1]

# Function to convert string tuple to numerical tuple
function parse_tuple(t::String)
    # Parse the string into Julia code
    parsed = Meta.parse(t)

    # Extract and convert to a tuple of numbers
    # Use `Tuple` to ensure it's a tuple and `Float64` for numerical conversion
    return Tuple(Float64.(eval(parsed)))
end
# Sampled parameter set
lsp = [parse_tuple(x) for x in lsp]

# Construct prediction interval
t = 300
@time lb,ub=predicreal(lsp,t)
lb = Float64.(lb)
ub = Float64.(ub)
x = 1:δ:L+1
f = plot(x, lb.*J, lw=0, fillrange=ub.*J, fillalpha=0.40, xlims = (0,200),ylims =(-5,25),color=:red, label=false, grid=false,tickfontsize = 20,margin = 10mm,framestyle=:box,
xticks=([1,50,100,150,200], [latexstring("1"),"",latexstring("100"),"",latexstring("200")]),yticks=([0,5,10,15,20], [latexstring("0"),"","","",latexstring("20")]),linecolor=:transparent)
f = plot!(x,Cmle.*J,color=:red,legend=false,lw=4,ls=:dash)
f = scatter!(1:1:200,data1, markersize = 4, markerstrokecolor=:blue, color=:blue)
f = hline!([0,20],lw=4,ls=:dash,color=:black)
f = annotate!(100, -11, text(L"i ", :left, 20))
f = annotate!(-200*0.17, 10, text(L"C_{1}^{\mathrm{o}} ", :left, 20))
savefig(f,"$path\\Figure4a.pdf")
display(f)
println("realisation Prediction Complete---------------------------------------------------------------")

# Prediction interval coverage.
flb = linear_interpolation(x,vec(lb.*J));
fub = linear_interpolation(x,vec(ub.*J));

global o = 0

for i = 1:1:200
    if data1[i] > fub(i) || data1[i] < flb(i)
        global o = o + 1
    end
end
out_rate = (o)/200* 100
println("The number of data outside interval for subpopulation 1: ",o)



println("The percentage of data outside interval: $out_rate %")

