using Plots, DifferentialEquations
using CSV
using DataFrames
using LaTeXStrings
using Interpolations
using Distributions
using Measures
using NLopt
using FilePathsBase

# In this script, we generate the result for multinomial measurement error model for Case 2

# Get the path of the current script
path = dirname(@__FILE__)

# Read count data from the CSV file.
data = DataFrame(CSV.File("$path\\data.csv"));
# Count data for subpopulation 1 and 2 in Case 2.
# Here, data1 corresponds to C_1^{o} and data2 corresponds to C_2^{o}.
data1 = data.a
data2 = data.b
# Fixed parameters
# Data is collected at time t = t1.
t1 = 1000
# The PDE in Equation (7) is solved on 0<=x<=L with grid spacing δ.
δ=0.5;L=199; J=20


    
# diff!(): Discretized PDE in Equation (7).
function diff!(d_u, u, p, t)
    # ------------------------------------------------------------------------------------------------------------------
    # Input:
    # d_u: (matrix) d_u[1, :] is dc1/dt and d_u[2, :] is dc2/dt.
    # u: (matrix)  u[1,n] is c1(x_n) and u[2,n] is c2(x_n)
    # p: (vector) [D1, D2, v1, v2, δ].
    # t: (number) Time.
    # Output:
    # d_u: (matrix) d_u[1, :] is dc1/dt and d_u[2, :] is dc2/dt.
    # ------------------------------------------------------------------------------------------------------------------

    (D1,D2,v1,v2,δ) = p
    (S,N) = size(u)

    # Boundary at x = 0, associated with Equation (24).
    d_u[1,1] = (D1/(δ^2))*((1-u[1,2]-u[2,2])*(u[1,2]-u[1,1]) + u[1,2]*(u[2,2]+u[1,2]-u[2,1]-u[1,1])) - (v1/δ)*u[1,2]*(1-u[1,2]-u[2,2])
    d_u[2,1] = (D2/(δ^2))*((1-u[1,2]-u[2,2])*(u[2,2]-u[2,1]) + u[2,2]*(u[2,2]+u[1,2]-u[2,1]-u[1,1])) - (v2/δ)*u[2,2]*(1-u[1,2]-u[2,2])

    # Associated with Equation (25).
    for n in 2:N-1
        d_u[1,n] = (D1/(2*δ^2))*(2-u[1,n]-u[2,n]-u[1,n+1]-u[2,n+1])*(u[1,n+1]-u[1,n]) + (D1/(2*δ^2))*(u[1,n]+u[1,n+1])*(u[1,n+1]+u[2,n+1]-u[1,n]-u[2,n]) - (D1/(2*δ^2))*(2-u[1,n-1]-u[2,n-1]-u[1,n]-u[2,n])*(u[1,n]-u[1,n-1]) - (D1/(2*δ^2))*(u[1,n]+u[1,n-1])*(u[1,n]+u[2,n]-u[1,n-1]-u[2,n-1]) - (v1/(2*δ))*(u[1,n+1]*(1-u[1,n+1]- u[2,n+1]) - u[1,n-1]*(1-u[1,n-1]-u[2,n-1]))
        d_u[2,n] = (D2/(2*δ^2))*(2-u[1,n]-u[2,n]-u[1,n+1]-u[2,n+1])*(u[2,n+1]-u[2,n]) + (D2/(2*δ^2))*(u[2,n]+u[2,n+1])*(u[1,n+1]+u[2,n+1]-u[1,n]-u[2,n]) - (D2/(2*δ^2))*(2-u[1,n-1]-u[2,n-1]-u[1,n]-u[2,n])*(u[2,n]-u[2,n-1]) - (D2/(2*δ^2))*(u[2,n]+u[2,n-1])*(u[1,n]+u[2,n]-u[1,n-1]-u[2,n-1]) - (v2/(2*δ))*(u[2,n+1]*(1-u[1,n+1]- u[2,n+1]) - u[2,n-1]*(1-u[1,n-1]-u[2,n-1]))                  
    end
    
    # Boundary at x = 199, associated with Equation (26).
    d_u[1,N] =  (-D1/(δ^2))*(1-u[1,N-1]-u[2,N-1])*(u[1,N]-u[1,N-1]) - (D1/(δ^2))*u[1,N-1]*(u[2,N]+u[1,N]-u[2,N-1]-u[1,N-1]) + (v1/δ)*u[1,N-1]*(1-u[1,N-1]-u[2,N-1])
    d_u[2,N] =  (-D2/(δ^2))*(1-u[1,N-1]-u[2,N-1])*(u[2,N]-u[2,N-1]) - (D2/(δ^2))*u[2,N-1]*(u[2,N]+u[1,N]-u[2,N-1]-u[1,N-1]) + (v2/δ)*u[2,N-1]*(1-u[1,N-1]-u[2,N-1])

    return d_u
end

# pdesolver(): Solves the ODE in diff!().
function pdesolver(time,D1,D2,v1,v2,δ,L)
    # ------------------------------------------------------------------------------------------------------------------
    # Input:
    # Time: (number) Solve ODE/PDE at t = Time.
    # D1: (number) Diffusivity for agents in subpopulation 1.
    # D2: (number) Diffusivity for agents in subpopulation 2.
    # v1: (number) Drift velocity for agents in subpopulation 1.
    # v2: (number) Drift velocity for agents in subpopulation 2.
    # δ: (number) Grid spacing.
    # L: (number) ODE/PDE is solved on 0 <= x <= L.
    # Output:
    # c1, c2: (vector) Solution of ODE/PDE at time t = Time.
    # ------------------------------------------------------------------------------------------------------------------
    # Construct initial condition for PDE.
    # Total number of mesh points N.
    N = Int(L/δ + 1)
    # Initial condition u0.
    u0 = zeros(2,N)
    # Update initial condition u0: c1(x) = 1 at 79 <= x <= 119.
    for n=1:N
       xx = (n-1)*δ
       if xx >=79 && xx <= 119
          u0[1,n] = 1
       end
    end
    # Update initial condition u0: c2(x) = 0.5 at 0 <= x < 79 and 119 < x <= 199.
    for n=1:N
        xx = (n-1)*δ
        if xx < 79
           u0[2,n] = 0.5
        end
        if xx > 119
           u0[2,n] = 0.5
        end
    end

    # Return the inital conditon if ODE/PDE is solved at t=0
    if time == 0
        c1_0 = vec(u0[1,:]); c2_0 = vec(u0[2,:])
        return c1_0,c2_0
    end

    # Solve the PDE using Heun's method at t=Time 
    p=(D1,D2,v1,v2,δ)
    tspan=(0,time)
    prob=ODEProblem(diff!,u0,tspan,p)
    alg=Heun() 
    sol=solve(prob,alg,saveat=time);
    sol = sol[end]
    u1 = sol[1,:];  u2= sol[2,:]
    c1 = vec(u1); c2 = vec(u2)
    return c1,c2
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
    # c1, c2: (vector) Solution of the PDE at time t = Time.
    #------------------------------------------------------------------------------------------------------------------
    # Solve the PDE.

    c1,c2=pdesolver(t1,a[1],a[2],a[3],a[4],δ,L)
    return c1,c2
end


# error(): Log-likelihood function for the multinomial Measurement Error Model for Case 2.
function error(data1,data2,δ,lx,a) 
    #------------------------------------------------------------------------------------------------------------------
    # Input:
    # data1: (vector) Count data for subpopulation 1.
    # data2: (vector) Count data for subpopulation 2.
    # δ: (number) Grid spacing.
    # L: (number) PDE solved on 0 <= x <= L.
    # a: (vector) Parameter vector.
    # Output:
    # e: (number) Log-likelihood for the multinomial Measurement Error Model (Case 2) with parameter vector a.
    #------------------------------------------------------------------------------------------------------------------
    # Solve the PDE at time t1 with parameter vector a.
    c1,c2=model(t1,a,δ,L);
    # Find c1(x_i, t1) and c2(x_i, t1) for i = 1, 2, 3, ..., 200.
    xlocdata = 0:δ:L
    fc1= linear_interpolation(xlocdata,c1);
    fc2= linear_interpolation(xlocdata,c2);
    # Estimate the loglikelihood function.
    e=0.0;
    for i = 1:lx+1
        c1 = fc1(i-1); c2 = fc2(i-1) 
        e = e + log((c1^data1[i]) * (c2^data2[i]) * ((1-c1-c2)^(J-data1[i]-data2[i])))
    end 
    return e
end

# fun(): Evaluate the log-likelihood function as a function of the parameters a.
function fun(a)
    # ------------------------------------------------------------------------------------------------------------------
    # Input:
    # a: (vector) Parameter vector.
    # Output:
    # error(data1, data2, δ, lx, a): (number) Log-likelihood for the multinomial Measurement Error Model (Case 2) with parameter vector a.
    # ------------------------------------------------------------------------------------------------------------------
    return error(data1,data2,δ,L,a)
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
function sample_para(D1_iv,D2_iv,v1_iv,v2_iv,fmle,M)
    # ------------------------------------------------------------------------------------------------------------------
    # Input:
    # D1_iv, D2_iv, v1_iv, v2_iv: (vector) Intervals for D1, D2, v1 and v2 for rejection sampling.
    # fmle: (number) Log-likelihood at MLE.
    # M: (number) Number of parameter sets required within the 95% log-likelihood threshold.
    # Output:
    # sampledP: (vector) M parameter sets within the 95% log-likelihood threshold.
    # ------------------------------------------------------------------------------------------------------------------
    # Degree of freedom.
    df=4
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
        D2g=rand(Uniform(D2_iv[1],D2_iv[2]))
        v2g=rand(Uniform(v2_iv[1],v2_iv[2]))
        # Evaluate the log-likelihood function using sampled parameter sets.
        ll = fun([D1g,D2g,v1g,v2g])
        if ll-fmle >= llstar
            # Sampled parameter set within the 95% log-likelihood threshold.
            # Update number of sampled parameter sets within the 95% log-likelihood threshold.
            count = count + 1
            # Update sampled parameter sets within the 95% log-likelihood threshold.
            push!(sampledP,(D1g,D2g,v1g,v2g))
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
    # ------------------------------------------------------------------------------------------------------------------
    # Generate empty lower and upper bounds of prediction intervals for subpopulations 1 and 2.
    xx = 0:δ:L
    gp=length(xx)
    lb1 = ones(gp); lb2 = ones(gp);
    ub1 = zeros(gp); ub2 = zeros(gp);
    for i = 1:length(lsp)
        # Solve the PDE with parameter set a.
        a = lsp[i]
        u1,u2 = model(t,[a[1],a[2],a[3],a[4]],δ,L)
        p1 = linear_interpolation(xx,vec(u1));
        p2 = linear_interpolation(xx,vec(u2));
        
        # Construct prediction interval for data realizations.
        for j = 1:length(xx)
            ulb1 = (quantile(Binomial(J,p1(xx[j])),[.05,.95])[1])/J
            uub1 = (quantile(Binomial(J,p1(xx[j])),[.05,.95])[2])/J
            ulb2 = (quantile(Binomial(J,p2(xx[j])),[.05,.95])[1])/J
            uub2 = (quantile(Binomial(J,p2(xx[j])),[.05,.95])[2])/J
 

            if ulb1 < lb1[j] 
                lb1[j] = ulb1
            end

            if ulb2 < lb2[j] 
                lb2[j] = ulb2
            end

            if uub1 > ub1[j] 
                ub1[j] = uub1
            end
            if uub2 > ub2[j] 
                ub2[j] = uub2
            end
        end
    end
    return lb1,ub1,lb2,ub2
end

#Expected Parameters
P1 = 0.8; P2 = 1
ρ1 = 0.2;ρ2 = 0
D1 = (P1)/(4); v1 = (ρ1*P1)/(2)
D2 = (P2)/(4); v2 = (ρ2*P2)/(2)


#MLE----------------------------------------------------------------------------
# Inital guess.
D1g = 0.18; D2g = 0.2
v1g=  0.07; v2g=  0.001
θG = [D1g,D2g,v1g,v2g] # inital guess
lb=[0.01,0.01,-0.1,-0.1] #lower bound
ub=[0.4,0.4,0.1,0.1] #upper bound
# Call numerical optimization routine to give the vector of parameters xopt, and the maximum loglikelihood fopt.
@time (xopt,fopt)  = optimise(fun,θG,lb,ub)
fmle=fopt
# Print MLE parameters
D1mle=xopt[1]; 
D2mle=xopt[2]; 
v1mle=xopt[3];
v2mle=xopt[4];
println("D1mle: ", D1mle)
println("D2mle: ", D2mle)
println("v1mle: ", v1mle)
println("v2mle: ", v2mle)
# Solve the PDE with MLE parameters
Cmle = model(t1,[D1mle,D2mle,v1mle,v2mle],δ,L)
println("MLE Complete---------------------------------------------------------------")


# Profile D1-------------------------------------------------------------------
df = 1
llstar = -quantile(Chisq(df),0.95)/2 # 95% asymptotic threshold
nptss=40 # points on the profile
D1min=0.1 # lower bound for the profile 
D1max=0.25 # upper bound for the profile 
D1range=LinRange(D1min,D1max,nptss) # vector of D1 values along the profile
nrange=zeros(3,nptss) # matrix to store the nuisance parameters once optimized out
llD1=zeros(nptss) # loglikelihood at each point along the profile
nllD1=zeros(nptss) # normalised loglikelihood at each point along the profile

@time for i in 1:nptss
    function fun1(aa) # function to return loglikelihood by fixing the interest parameter along the profile
        return error(data1,data2,δ,L,[D1range[i],aa[1],aa[2],aa[3]])
    end
    local lb1=[0.01,-0.1,-0.1] # lower bound 
    local ub1=[0.25,0.1,0.1] # upper bound
    local θG1=[D2mle,v1mle,v2mle] # initial estimate - take the MLE 
    @time local (xo,fo)=optimise(fun1,θG1,lb1,ub1)
    nrange[:,i]=xo[:] # store the nuisance parameters
    llD1[i]=fo[1] # store the loglikelihood 
    display(i)
end
nllD1=llD1.-maximum(llD1); # calculate normalised loglikelihood
println("Profile D1 Complete-----------------------------------------------------------------------------")
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

# Profile D2-------------------------------------------------------------------
df = 1
llstar = -quantile(Chisq(df),0.95)/2 # 95% asymptotic threshold
nptss=40 # points on the profile
D2min=0.05 # lower bound for the profile 
D2max=0.2 # upper bound for the profile 
D2range=LinRange(D2min,D2max,nptss) # vector of D2 values along the profile
nrange=zeros(3,nptss) # matrix to store the nuisance parameters once optimized out
llD2=zeros(nptss) # loglikelihood at each point along the profile
nllD2=zeros(nptss) # normalised loglikelihood at each point along the profile

@time for i in 1:nptss
    function fun2(aa) # function to return loglikelihood by fixing the interest parameter along the profile
        return error(data1,data2,δ,L,[aa[1],D2range[i],aa[2],aa[3]])
    end
    local lb2=[0.01,-0.1,-0.1] # lower bound 
    local ub2=[0.4,0.1,0.1] # upper bound
    local θG2=[D1mle,v1mle,v2mle] # initial estimate - take the MLE 
    @time local (xo,fo)=optimise(fun2,θG2,lb2,ub2)
    nrange[:,i]=xo[:] # store the nuisance parameters
    llD2[i]=fo[1] # store the loglikelihood 
    display(i)
end
nllD2=llD2.-maximum(llD2); # calculate normalised loglikelihood
println("Profile D2 Complete-----------------------------------------------------------------------------")
# Store the loglikelihood in csv.
df = DataFrame(Value = llD2)
CSV.write("$path\\llD2.csv", df)

# Plot the univariate profile likelihood for D2 with D1.
df = CSV.read("$path\\llD2.csv", DataFrames.DataFrame)
llD2 = df[:, 1]
nllD2=llD2.-maximum(llD2)
s1=plot!(D2range,nllD2,lw=6,xlim=(0,0.4),ylim=(-3,0.1),legend=false,tickfontsize=20,labelfontsize=20,margin = 10mm,grid=false,
xticks=([0,0.1,0.2,0.3,0.4], [latexstring("0"),"", latexstring("0.2"),"",latexstring("0.4")]),yticks=([-3,-2,-1,0],[latexstring("-3"),latexstring("-2"),latexstring("-1"),latexstring("0")]),framestyle=:box,color=1,line=:dot,left_margin = 15mm)
s1=vline!([D2mle],lw=6,color=3,line=:dot)
s1=hline!([llstar],lw=6,color=2)
s1 = annotate!(0.17, -3.6, text(L"D_{1}, D_{2} ", :left, 20))
s1 = annotate!(-0.09, -1.5, text(L"\bar{\ell}_{p}", :left, 20))
savefig(s1,"$path\\Figure5b.pdf")
display(s1) 


# Profile v1-------------------------------------------------------------------
df = 1
llstar = -quantile(Chisq(df),0.95)/2 # 95% asymptotic threshold
nptss=40 # points on the profile
v1min= 0.05 # lower bound for the profile  
v1max=0.09 # upper bound for the profile  
v1range=LinRange(v1min,v1max,nptss) # vector of v1 values along the profile
nrange=zeros(3,nptss) # matrix to store the nuisance parameters once optimized out
llv1=zeros(nptss) # loglikelihood at each point along the profile
nllv1=zeros(nptss) # normalised loglikelihood at each point along the profile

@time for i in 1:nptss
    function fun3(aa) # function to return loglikelihood by fixing the interest parameter along the profile
        return error(data1,data2,δ,L,[aa[1],aa[2],v1range[i],aa[3]])
    end
    local lb3=[0.01,0.01,-0.1] # lower bound 
    local ub3=[0.25,0.25,0.1] # upper bound
    local θG3=[D1mle,D2mle,v2mle] # initial estimate - take the MLE 
    @time local (xo,fo)=optimise(fun3,θG3,lb3,ub3)
    nrange[:,i]=xo[:] # store the nuisance parameters
    llv1[i]=fo[1] # store the loglikelihood 
    display(i)
end
nllv1=llv1.-maximum(llv1); # calculate normalised loglikelihood
println("Profile v1 Complete-----------------------------------------------------------------------------")
# Store the loglikelihood in csv.
df = DataFrame(Value = llv1)
CSV.write("$path\\llv1.csv", df)
# Plot the univariate profile likelihood for v1 and hold.
df = CSV.read("$path\\llv1.csv", DataFrames.DataFrame)
llv1 = df[:, 1]
nllv1=llv1.-maximum(llv1)
s2=plot(v1range,nllv1,lw=6,xlim=(-0.002,0.1),ylim=(-3,0.1),legend=false,tickfontsize=20,labelfontsize=20,margin = 10mm,grid=false,
xticks=([0,0.05,0.1], [latexstring("0"),latexstring("0.05"),latexstring("0.1")]),yticks=([-3,-2,-1,0],[latexstring("-3"),latexstring("-2"),latexstring("-1"),latexstring("0")]),framestyle=:box,left_margin = 15mm)
s2=hline!([llstar],lw=6)
s2=vline!([v1mle],lw=6)

# Profile v2-------------------------------------------------------------------
df = 1
llstar = -quantile(Chisq(df),0.95)/2 # 95% asymptotic threshold
nptss=40 # points on the profile
v2min=-0.01 # lower bound for the profile 
v2max=0.01 # upper bound for the profile 
v2range=LinRange(v2min,v2max,nptss) # vector of v2 values along the profile
nrange=zeros(3,nptss) # matrix to store the nuisance parameters once optimized out
llv2=zeros(nptss) # loglikelihood at each point along the profile
nllv2=zeros(nptss) # normalised loglikelihood at each point along the profile

@time for i in 1:nptss
    function fun4(aa) # function to return loglikelihood by fixing the interest parameter along the profile
        return error(data1,data2,δ,L,[aa[1],aa[2],aa[3],v2range[i]])
    end
    local lb4=[0.01,0.01,-0.1] # lower bound 
    local ub4=[0.25,0.25,0.1] # upper bound
    local θG4=[D1mle,D2mle,v1mle] # initial estimate - take the MLE 
    @time local (xo,fo)=optimise(fun4,θG4,lb4,ub4)
    nrange[:,i]=xo[:] # store the nuisance parameters
    llv2[i]=fo[1] # store the loglikelihood 
    display(i)
end
nllv2=llv2.-maximum(llv2); # calculate normalised loglikelihood
println("Profile v2 Complete-----------------------------------------------------------------------------")
# Store the loglikelihood in csv.
df = DataFrame(Value = llv2)
CSV.write("$path\\llv2.csv", df)

# Plot the univariate profile likelihood for v2 with v1.
df = CSV.read("$path\\llv2.csv", DataFrames.DataFrame)
llv2 = df[:, 1]
nllv2=llv2.-maximum(llv2)
s2=plot!(v2range,nllv2,lw=6,xlim=(-0.002,0.1),ylim=(-3,0.1),legend=false,tickfontsize=20,labelfontsize=20,margin = 10mm,grid=false,
xticks=([0,0.05,0.1], [latexstring("0"),latexstring("0.05"),latexstring("0.1")]),yticks=([-3,-2,-1,0],[latexstring("-3"),latexstring("-2"),latexstring("-1"),latexstring("0")]),framestyle=:box,color=1, line=:dot,left_margin = 15mm)
s2=vline!([v2mle],lw=6,color=3,line=:dot)
s2=hline!([llstar],lw=6,color=2)
s2 = annotate!(0.044, -3.6, text(L"v_{1}, v_{2} ", :left, 20))
s2 = annotate!((-0.090/4)-0.0025, -1.5, text(L"\bar{\ell}_{p}", :left, 20))
savefig(s2,"$path\\Figure5d.pdf")
display(s2) 

# Sample the parameters from rejection sampling.
# -------------------------------------------------
# -------------------------------------------------
# Notice: If the parameters have already been sampled and stored in lsp.csv, the prediction interval can be constructed directly using lsp.csv.
# -------------------------------------------------
# -------------------------------------------------
# sample the parameters
D1_iv = [0.06,0.25]
D2_iv = [0.02,0.17]
v1_iv = [0.05,0.08]
v2_iv = [-0.003,0.007]
M = 500
@time lsp =  sample_para(D1_iv,D2_iv,v1_iv,v2_iv,fmle,M)
# Store sampled parameter set in lsp.csv
df = DataFrame(Value = lsp)
CSV.write("$path\\lsp.csv", df)

# Plot the parameters and their lower and upper bounds for rejection sampling.
D1sampled = [Para[1] for Para in lsp]
D2sampled = [Para[2] for Para in lsp]
v1sampled = [Para[3] for Para in lsp]
v2sampled = [Para[4] for Para in lsp]
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

q3=scatter(v1sampled,legend=false,grid=false,xticks=([1,500],["1","500"]),tickfontsize = 15,size=(600, 200),markersize = 3,
ylims = (v1_iv[1]-0.01,v1_iv[2]+0.01),yticks=(v1_iv,  [ string(v1_iv[1]), string(v1_iv[2])]),markerstrokecolor = :white, color=:black)
q3=hline!(v1_iv,legend=false,lw = 4, linestyle = :dash, color=:blue)
savefig(q3,"$path\\inval_q1.pdf")
display(q3)

q4=scatter(v2sampled,legend=false,grid=false,xticks=([1,500],["1","500"]),tickfontsize = 15,size=(600, 200),markersize = 3,
ylims = (v2_iv[1]-0.005,v2_iv[2]+0.005),yticks=(v2_iv ,  [ string(v2_iv[1]), string(v2_iv[2])]),markerstrokecolor = :white, color=:black)
q4=hline!(v2_iv,legend=false,lw = 4, linestyle = :dash, color=:blue)
savefig(q4,"$path\\inval_q2.pdf")
display(q4)

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
t = 1000
@time lb1,ub1,lb2,ub2=predicreal(lsp,t)
lb = Float64.(lb); lb1 = Float64.(lb1); lb2 = Float64.(lb2);
ub = Float64.(ub); ub1 = Float64.(ub1); ub2 = Float64.(ub2);
x = 1:δ:L+1

f1 = plot(x, lb1.*J, lw=0, fillrange=ub1.*J, fillalpha=0.40, xlims = (0,200),ylims =(-5,25),color=:red, label=false, grid=false,tickfontsize = 20,margin = 10mm,framestyle=:box,
xticks=([1,50,100,150,200], [latexstring("1"),"",latexstring("100"),"",latexstring("200")]),yticks=([0,5,10,15,20], [latexstring("0"),"","","",latexstring("20")]),linecolor=:transparent)
f1 = plot!(x,Cmle[1].*J,color=:red,legend=false,lw=4,ls=:dash)
f1 = scatter!(1:1:200,data1, markersize = 4, markerstrokecolor=:blue, color=:blue)
f1 = hline!([0,20],legend=false,lw = 4, linestyle = :dash, color=:black)
f1 = annotate!(100, -10, text(L"i ", :left, 20))
f1 = annotate!(-200*0.17, 11, text(L"C_{1}^{\mathrm{o}} ", :left, 20))
savefig(f1,"$path\\Figure6b.pdf")
display(f1)

f2 = plot(x, lb2.*J, lw=0, fillrange=ub2.*J, fillalpha=0.40, xlims = (0,200),ylims =(-5,25),color=:green, label=false, grid=false,tickfontsize = 20,margin = 10mm,framestyle=:box,
xticks=([1,50,100,150,200], [latexstring("1"),"",latexstring("100"),"",latexstring("200")]),yticks=([0,5,10,15,20], [latexstring("0"),"","","",latexstring("20")]),linecolor=:transparent)
f2 = plot!(x,Cmle[2].*J,color=:green,legend=false,lw=4,ls=:dash)
f2 = scatter!(1:1:200,data2, markersize = 4, markerstrokecolor=:blue, color=:blue)
f2 = hline!([0,20],legend=false,lw = 4, linestyle = :dash, color=:black)
f2 = annotate!(100, -11, text(L"i ", :left, 20))
f2 = annotate!(-200*0.17, 10, text(L"C_{2}^{\mathrm{o}} ", :left, 20))
savefig(f2,"$path\\Figure6d.pdf")
display(f2)
println("realisation Prediction Complete---------------------------------------------------------------")

# Prediction interval coverage.
flb1 = linear_interpolation(x,vec(lb1.*J));
fub1 = linear_interpolation(x,vec(ub1.*J));
flb2 = linear_interpolation(x,vec(lb2.*J));
fub2 = linear_interpolation(x,vec(ub2.*J));
global o1 = 0
global o2 = 0
for i = 1:1:200
    if data1[i] > fub1(i) || data1[i] < flb1(i)
        global o1 = o1 + 1
    end

    if data2[i] > fub2(i) || data2[i] < flb2(i)
        global o2 = o2 + 1
    end
end
out_rate = (o1+o2)/400* 100
println("The number of data outside interval for subpopulation 1: ",o1)
println("The number of data outside interval for subpopulation 2: ",o2)

println("The percentage of data outside interval: $out_rate %")