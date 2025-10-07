using Plots
using Measures
using CSV
using DataFrames
using FilePathsBase
using LaTeXStrings
# In this script, we create Figure  2(a)-(d).

# Get the path of the current script.
path = dirname(@__FILE__)
t=300

# Read the CSV file: indices for each agent in subpopulations 1 at the initial condition.
index_0 = DataFrame(CSV.File("$path\\index_0.csv")); 


# Read the CSV file: count data for subpopulations 1 at the initial condition.
data_0 = DataFrame(CSV.File("$path\\data_0.csv")); 
C_1_0 = data_0.a;

# Read the CSV file: indices for each agent in subpopulations 1 at time t.
index = DataFrame(CSV.File("$path\\index.csv")); 


# Read the CSV file: count data for subpopulations 1 at time t.
data = DataFrame(CSV.File("$path\\data.csv")); 
C_1 = data.a;


# Plot the snapshots for subpopulations 1 at the initial condition.
f1 = scatter(index_0.i, index_0.j, label=false, marker=:circle,
            xlims = [1,200],ylims = [1,20],xticks=([1,50,100,150,200], [latexstring("1"),"","","",latexstring("200")]),
            yticks=([1,10,20],[latexstring("1"),"",latexstring("20")]),legend = false,markersize=1,
            grid=false,tickfontsize = 20,aspect_ratio = :equal,
            color=:red,markerstrokecolor=:red,framestyle=:box)
f1 = annotate!(5, 20+7, text(L"k = " * latexstring(string(0)), :left, 20))
f1 = annotate!(200/2, -10, text(L"i ", :left, 20))
f1 = annotate!(-35, 20/2, text(L"j ", :left, 20))

# Plot the snapshots for subpopulations 1 at time t.
f2 = scatter(index.i, index.j, label=false, marker=:circle,
            xlims = [1,200],ylims = [1,20],xticks=([1,50,100,150,200], [latexstring("1"),"","","",latexstring("200")]),
            yticks=([1,10,20],[latexstring("1"),"",latexstring("20")]),legend = false,markersize=1,
            grid=false,tickfontsize = 20,aspect_ratio = :equal,
            color=:red,markerstrokecolor=:red,framestyle=:box)
f2 = annotate!(5, 20+7, text(L"k = " * latexstring(string(t)), :left, 20))
f2 = annotate!(200/2, -13, text(L"i ", :left, 20))
f2 = annotate!(-35,  20/2, text(L"j ", :left, 20))

# Plot the count data for subpopulation 1 at the initial condition.
f3 = scatter(1:1:200,C_1_0,xticks=([1,50,100,150,200], [latexstring("1"),"","","",latexstring("200")]),
            yticks=([-5,0,5,10,15,20,25],["",latexstring("0"),"","","",latexstring("20"),""]),label=false,xlims = [1,200],ylims = [-5,25],grid = false,
        tickfontsize = 20,color=:red, framestyle=:box,markerstrokecolor=:red,aspect_ratio = :equal)
f3 = annotate!(200/2, -18, text(L"i ", :left, 20))
f3 = annotate!(-35,  25/2-4, text(L"C_{1}^{\mathrm{o}}", :left, 15))
f3 = annotate!(5, 20+15, text(L"k = " * latexstring(string(0)), :left, 20))
f3 = hline!([0,20],legend=false,lw = 4, linestyle = :dash, color=:black)

# Plot the count data for subpopulation 1 at time t.
f4 = scatter(1:1:200,C_1,xticks=([1,50,100,150,200], [latexstring("1"),"","","",latexstring("200")]),
            yticks=([-5,0,5,10,15,20,25],["",latexstring("0"),"","","",latexstring("20"),""]),label=false,xlims = [1,200],ylims = [-5,25],grid = false,
        tickfontsize = 20,color=:red, framestyle=:box,markerstrokecolor=:red,aspect_ratio = :equal)
f4 = annotate!(200/2, -18, text(L"i ", :left, 20))
f4 = annotate!(-35,  25/2-4, text(L"C_{1}^{\mathrm{o}}", :left, 15))
f4 = annotate!(5, 20+15, text(L"k = " * latexstring(string(t)), :left, 20))
f4 = hline!([0,20],legend=false,lw = 4, linestyle = :dash, color=:black)

f = plot(f1, f2, f3, f4,layout = (2,2), link = :x, top_margin = 0mm, right_margin = 5mm,left_margin = 15mm,size=(1200,320))  # Here we set the top margin
savefig(f,"$path\\Figure2a-d.pdf")
display(f)