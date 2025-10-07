using Plots
using Measures
using CSV
using DataFrames
using FilePathsBase
using LaTeXStrings

# In this script we genrate Figure 3(e)-(j).

# Get the path of the current script.
path = dirname(@__FILE__)
t=2000

# Read the CSV file: indices for each agent in subpopulations 1, 2 and 3 at the initial condition.
index1_0 = DataFrame(CSV.File("$path\\index1_0.csv")); 
index2_0 = DataFrame(CSV.File("$path\\index2_0.csv")); 
index3_0 = DataFrame(CSV.File("$path\\index3_0.csv")); 

# Read the CSV file: count data for subpopulations 1, 2 and 3 at the initial condition.
data_0 = DataFrame(CSV.File("$path\\data_0.csv")); 
C_1_0 = data_0.a; C_2_0 = data_0.b; C_3_0 = data_0.c; 

# Read the CSV file: indices for each agent in subpopulations 1, 2 and 3 at time t.
index1 = DataFrame(CSV.File("$path\\index1.csv")); 
index2 = DataFrame(CSV.File("$path\\index2.csv")); 
index3 = DataFrame(CSV.File("$path\\index3.csv")); 

# Read the CSV file: count data for subpopulations 1, 2 and 3 at time t.
data = DataFrame(CSV.File("$path\\data.csv")); 
C_1 = data.a; C_2 = data.b; C_3 = data.c;  


# Plot the snapshots for subpopulations 1, 2 and 3 at the initial condition.
f1 = scatter(index1_0.i, index1_0.j, label=false, marker=:circle,
            xlims = [1,200],ylims = [1,20],xticks=([1,50,100,150,200], [latexstring("1"),"","","",latexstring("200")]),
            yticks=([1,10,20],[latexstring("1"),"",latexstring("20")]),legend = false,markersize=1,
            grid=false,tickfontsize = 20,aspect_ratio = :equal,
            color=:red,markerstrokecolor=:red,framestyle=:box,right_margin=5mm,bottom_margin=5mm)
f1 = scatter!(index2_0.i, index2_0.j, label=false, marker=:circle,color=:blue,markerstrokecolor=:blue,markersize=1)
f1 = scatter!(index3_0.i, index3_0.j, label=false, marker=:circle,color=:green,markerstrokecolor=:green,markersize=1)        
f1 = annotate!(5, 20+7, text(L"k = " * latexstring(string(0)), :left, 20))
f1 = annotate!(200/2, -10, text(L"i ", :left, 20))
f1 = annotate!(-35, 20/2, text(L"j ", :left, 20))

# Plot the snapshots for subpopulations 1, 2 and 3 at time t.
f2 = scatter(index1.i, index1.j, label=false, marker=:circle,
            xlims = [1,200],ylims = [1,20],xticks=([1,50,100,150,200], [latexstring("1"),"","","",latexstring("200")]),
            yticks=([1,10,20],[latexstring("1"),"",latexstring("20")]),legend = false,markersize=1,
            grid=false,tickfontsize = 20,aspect_ratio = :equal,
            color=:red,markerstrokecolor=:red,framestyle=:box)
f2 = scatter!(index2.i, index2.j, label=false, marker=:circle,color=:blue,markerstrokecolor=:blue,markersize=1,right_margin=5mm) 
f2 = scatter!(index3.i, index3.j, label=false, marker=:circle,color=:green,markerstrokecolor=:green,markersize=1,right_margin=5mm) 
f2 = annotate!(5, 20+7, text(L"k = " * latexstring(string(t)), :left, 20))
f2 = annotate!(200/2, -13, text(L"i ", :left, 20))
f2 = annotate!(-35,  20/2, text(L"j ", :left, 20))

# Plot the count data for subpopulation 1 at the initial condition.
f3 = scatter(1:1:200,C_1_0,xticks=([1,50,100,150,200], [latexstring("1"),"","","",latexstring("200")]),
            yticks=([-5,0,5,10,15,20,25],["",latexstring("0"),"","","",latexstring("20"),""]),label=false,xlims = [1,200],ylims = [-5,25],grid = false,
        tickfontsize = 20,color=:red, framestyle=:box,markerstrokecolor=:red,aspect_ratio = :equal)
f3 = annotate!(200/2, -18, text(L"i ", :left, 20))
f3 = annotate!(-40,  25/2-4, text(L"C_{1}^{\mathrm{o}}", :left, 15))
f3 = annotate!(5, 20+15, text(L"k = " * latexstring(string(0)), :left, 20))
f3 = hline!([0,20],legend=false,lw = 4, linestyle = :dash, color=:black)

# Plot the count data for subpopulation 2 at the initial condition.
f4 = scatter(1:1:200,C_2_0,xticks=([1,50,100,150,200], [latexstring("1"),"","","",latexstring("200")]),
            yticks=([-5,0,5,10,15,20,25],["",latexstring("0"),"","","",latexstring("20"),""]),label=false,xlims = [1,200],ylims = [-5,25],grid = false,
        tickfontsize = 20,color=:blue, framestyle=:box,markerstrokecolor=:blue,aspect_ratio = :equal)
f4 = annotate!(200/2, -18, text(L"i ", :left, 20))
f4 = annotate!(-40,  25/2-4, text(L"C_{2}^{\mathrm{o}}", :left, 15))
f4 = annotate!(5, 20+15, text(L"k = " * latexstring(string(0)), :left, 20))
f4 = hline!([0,20],legend=false,lw = 4, linestyle = :dash, color=:black)

# Plot the count data for subpopulation 3 at the initial condition.
f5 = scatter(1:1:200,C_3_0,xticks=([1,50,100,150,200], [latexstring("1"),"","","",latexstring("200")]),
            yticks=([-5,0,5,10,15,20,25],["",latexstring("0"),"","","",latexstring("20"),""]),label=false,xlims = [1,200],ylims = [-5,25],grid = false,
        tickfontsize = 20,color=:green, framestyle=:box,markerstrokecolor=:green,aspect_ratio = :equal)
f5 = annotate!(200/2, -18, text(L"i ", :left, 20))
f5 = annotate!(-40,  25/2-4, text(L"C_{3}^{\mathrm{o}}", :left, 15))
f5 = annotate!(5, 20+15, text(L"k = " * latexstring(string(0)), :left, 20))
f5 = hline!([0,20],legend=false,lw = 4, linestyle = :dash, color=:black)

# Plot the count data for subpopulation 1 at time t.
f6 = scatter(1:1:200,C_1,xticks=([1,50,100,150,200], [latexstring("1"),"","","",latexstring("200")]),
            yticks=([-5,0,5,10,15,20,25],["",latexstring("0"),"","","",latexstring("20"),""]),label=false,xlims = [1,200],ylims = [-5,25],grid = false,
        tickfontsize = 20,color=:red, framestyle=:box,markerstrokecolor=:red,aspect_ratio = :equal)
f6 = annotate!(200/2, -18, text(L"i ", :left, 20))
f6 = annotate!(-40,  25/2-4, text(L"C_{1}^{\mathrm{o}}", :left, 15))
f6 = annotate!(5, 20+15, text(L"k = " * latexstring(string(t)), :left, 20))
f6 = hline!([0,20],legend=false,lw = 4, linestyle = :dash, color=:black)


# Plot the count data for subpopulation 2 at time t.
f7 = scatter(1:1:200,C_2,xticks=([1,50,100,150,200], [latexstring("1"),"","","",latexstring("200")]),
            yticks=([-5,0,5,10,15,20,25],["",latexstring("0"),"","","",latexstring("20"),""]),label=false,xlims = [1,200],ylims = [-5,25],grid = false,
        tickfontsize = 20,color=:blue, framestyle=:box,markerstrokecolor=:blue,aspect_ratio = :equal)
f7 = annotate!(200/2, -18, text(L"i ", :left, 20))
f7 = annotate!(-40,  25/2-4, text(L"C_{2}^{\mathrm{o}}", :left, 15))
f7 = annotate!(5, 20+15, text(L"k = " * latexstring(string(t)), :left, 20))
f7 = hline!([0,20],legend=false,lw = 4, linestyle = :dash, color=:black)


# Plot the count data for subpopulation 3 at time t.
f8 = scatter(1:1:200,C_3,xticks=([1,50,100,150,200], [latexstring("1"),"","","",latexstring("200")]),
            yticks=([-5,0,5,10,15,20,25],["",latexstring("0"),"","","",latexstring("20"),""]),label=false,xlims = [1,200],ylims = [-5,25],grid = false,
        tickfontsize = 20,color=:green, framestyle=:box,markerstrokecolor=:green,aspect_ratio = :equal)
f8 = annotate!(200/2, -18, text(L"i ", :left, 20))
f8 = annotate!(-40,  25/2-4, text(L"C_{3}^{\mathrm{o}}", :left, 15))
f8 = annotate!(5, 20+15, text(L"k = " * latexstring(string(t)), :left, 20))
f8 = hline!([0,20],legend=false,lw = 4, linestyle = :dash, color=:black)


f = plot(f1,f2,f3,f6,f4,f7,f5,f8, layout = (4,2), link = :x, top_margin = 0mm, right_margin = 5mm,left_margin = 15mm,size=(990,480))  # Here we set the top margin
savefig(f,"$path\\FigureS6.pdf")
display(f)