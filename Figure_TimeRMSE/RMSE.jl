begin
    #using Zygote
    using SciMLSensitivity
    using OrdinaryDiffEq
    using Random
    using Setfield
    using Plots
    using DataDrivenDiffEq, DataDrivenSparse
    using Statistics, StatsBase
    using Printf
    using Latexify
    using Flux
    using JLD2
    using Catalyst, ModelingToolkit
    using ComponentArrays
    using DataInterpolations
    using Zygote
    using CSV, DataFrames
    using Statistics
    using LinearAlgebra
    using StatsBase
    using LaTeXStrings
end


using CUDA
CUDA.allowscalar(false)
CUDA.memory_status() 
cd(Base.source_path()*"/..")
pwd()
rmse_ozone = []
for i_latent_species in 2:10
open("results/$(i_latent_species)/simady/rmse_test.txt", "r") do file
    for line in eachline(file)
        first_float = match(r"\d+\.\d+", line)
        if first_float !== nothing
            push!(rmse_ozone, parse(Float64, first_float.match))
            break 
        end
    end
end

end
rmse_ozone = rmse_ozone./36.19 .* 100
plot_rmse = plot(rmse_ozone,
        marker = (:circle,5),
        #label = "testing set",
        legend = :none,
        xlabel = "Number of latent species",
        ylabel = "RMSE (ppb)",
        color = :blue,
        xticks = (1:9, 2:10),
        titlefontsize = 17,
        xtickfontsize = 14, ytickfontsize = 14, 
        xlabelfontsize = 16, ylabelfontsize = 16,
        legendfontsize = 16,grid = true,
        size = (700,450),
        left_margin = 3Plots.mm,
        right_margin = 5Plots.mm,
        bottom_margin = 2Plots.mm
)


times_mcm = []
for i in 0:9
    i = @sprintf("%04d", i)
filename = "scenarios_timing/scenario_$(i)/run_init.log"
lines = readlines(filename)

elapsed_lines = filter(line -> startswith(line, " Elapsed time for the MCM integration step:"), lines)

times = sum([parse(Float64, match(r"([0-9.]+[Ee]?-?[0-9]*)", line).match) for line in elapsed_lines][end-14400:end])

    push!(times_mcm,times)
end
times_mcm
mean(times_mcm)

cpu_time = []
gpu_time = []

for i_latent_species in 2:10

    path = "results/$(i_latent_species)/timing/"
    JLD2.@load path*"bench_single.jld" bench_cpu_vec bench_gpu_vec
    cpu_time_temp = []
    gpu_time_temp = []
for j in 1:size(bench_cpu_vec)[1]
    push!(cpu_time_temp, mean(bench_cpu_vec[j].times)./1e9)
    push!(gpu_time_temp, mean(bench_gpu_vec[j].times)./1e9)
end
push!(cpu_time, mean(cpu_time_temp))
  push!(gpu_time, mean(gpu_time_temp))  
end

cpu_time = log10.(cpu_time)
gpu_time = log10.(gpu_time)
mean(gpu_time)
rmse_ozone = []
for i_latent_species in 2:10

open("results/$(i_latent_species)/plots/rmse_test.txt", "r") do file
    for line in eachline(file)
        first_float = match(r"\d+\.\d+", line)
        if first_float !== nothing
            push!(rmse_ozone, parse(Float64, first_float.match))
            break 
        end
    end
end

end
rmse_ozone = rmse_ozone./36.19 .* 100

plot_timingA = plot(cpu_time,
        marker = (:square,8),
    linewidth = 2,
        label = "SIMADy (CPU)",
        xlabel = "Number of latent species",
        ylabel = "Computational time (second)",
        color = :red,
        xticks = (1:9, 2:10),
        yformatter=ytick -> "10^{$(Int(round(ytick)))}",
        ylim = (-2,3),
        titlefontsize = 17,
        xtickfontsize = 14, ytickfontsize = 14, 
        xlabelfontsize = 16, ylabelfontsize = 16,
        legendfontsize = 16,grid = true,
        size = (700,450),
        left_margin = 10Plots.mm,
        right_margin = 10Plots.mm,
        up_margin = 10Plots.mm,
        bottom_margin = 10Plots.mm,
        legend=:none
)
plot!(gpu_time,
        marker = (:utriangle,8),
        linewidth = 2,
        label = "SIMADy (GPU)",
        legend=:none,
        color = :red,
)
plot!(
    1:9,
    repeat([log10(mean(times_mcm))],9), 
    label="MCM", 
    marker = (:circle,8),
    color=:red,
    legend=:none,
    linewidth = 2,
)


plot!(twinx(), [1,2,3,4,5,6,7,8,9], rmse_ozone,
        marker = (:circle,8),
    linewidth = 2,
            titlefontsize = 17,
    color = :blue,
    ylim = (8,11,0.5),
        xtickfontsize = 14, ytickfontsize = 14, 
        xlabelfontsize = 16, ylabelfontsize = 16,
        legendfontsize = 16,grid = true,
        legend=:none,
        ylabel = "Ozone error (%)",
)

savefig(plot_timingA, "timingA.svg")

mean_flag =  [1, 1, 1, 1,  1,  1,  1,   1,   1,   1,    1,    0,    0,     0,     0]
scale_case = [1,  10,  100,  1000,  10000, 100000]



begin
    cpu_time_vec = []
    for (i, case) in enumerate(scale_case)
        cpu_time_temp_mean = []
            JLD2.@load "code/cpu/$(case)/timing/bench_multiple_$(case).jld" bench_cpu
            push!(cpu_time_vec, minimum(bench_cpu.times))
    end
    cpu_time_vec = cpu_time_vec ./ 1e9
end
begin
    gpu_time_vec = []
    for (i, case) in enumerate(scale_case)
        gpu_time_temp_mean = []
            JLD2.@load "code/gpu/$(case)/timing/bench_multiple_$(case).jld" bench_gpu
            push!(gpu_time_vec, minimum(bench_gpu.times))
    end
    gpu_time_vec = gpu_time_vec ./ 1e9
end

cpu_time_vec[end]/gpu_time_vec[end]
gpu_time_vec
((mean(times_mcm).*scale_case) ./cpu_time_vec)[end]
((mean(times_mcm).*scale_case) ./gpu_time_vec)[end]
x = Int.(range(0, 5, length=6) )
plot_timingB = plot(
        x,
        log10.(cpu_time_vec),
        marker = (:square,8),
        linewidth = 2,
        label = "SIMADy (CPU)",
        xlabel = "Number of simulations",
        ylabel = "Computational time (second)",
        color = :red,
        xticks=(0:1:5),
        yticks=-1:1:8,
        xformatter=xtick -> "10^{$(Int(round(xtick)))}",
        yformatter=ytick -> "10^{$(Int(round(ytick)))}",
        titlefontsize = 17,
        xtickfontsize = 10, ytickfontsize = 10, 
        xlabelfontsize = 16, ylabelfontsize = 16,
        legendfontsize = 16,grid = true,
        size = (700,450),
        left_margin = 10Plots.mm,
        right_margin = 10Plots.mm,
        up_margin = 10Plots.mm,
        bottom_margin = 10Plots.mm
)

plot!(
        x,
        log10.(gpu_time_vec),
        marker = (:utriangle,8),
        linewidth = 2,
        label = "SIMADy (GPU)",
        color = :red,
)


plot!(
    x,
    log10.(mean(times_mcm).*scale_case),
    marker = (:circle,8),
    label="MCM", 
    linewidth = 2,
    color = :red,
)

savefig(plot_timingB, "timingB.svg")