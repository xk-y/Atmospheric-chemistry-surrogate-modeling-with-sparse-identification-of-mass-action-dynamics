


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
    using Clustering
end


    using CUDA
    CUDA.allowscalar(false)
    CUDA.memory_status() 
    cd(Base.source_path()*"/..")
ndays = 10

saveat = 60.0
dt = 60 # minutes
nruns = 1070
n_species = 5838
nmete = 5
ntimestep = 24*ndays +1
times = LinRange(0, saveat * (ndays * 24), ntimestep)
startspec = 1

n_latent_species = 6
seed=1234
Random.seed!(seed)
encoder = abs.(Flux.glorot_uniform(Random.seed!(seed), n_latent_species, n_species)) 
size_encoder = size(reshape(encoder,:))[1]
function encoder_(encoder,X_3d,batchsize)
    X_2d = reshape(X_3d,(n_species,:))
    x_2d = encoder * X_2d
    x_3d = reshape(x_2d,(n_latent_species, :, batchsize))
end

function decoder_(decoder,x_3d,batchsize)
    x_2d = reshape(x_3d,(n_latent_species,:))
    X_2d = decoder' * x_2d
    X_3d = reshape(X_2d,(n_species,:, batchsize))
end
n_latent_emit_species = n_latent_species


JLD2.@load "../c_test_utils.jld" ref_data_max ref_data_min dc_std
JLD2.@load "../../testing_set.jld" ref_data_test ref_emit_test ref_params_test specname



data = ref_data_test[10,:,:]
data_std = (data .- mean(data, dims=1)) ./ std(data, dims=1)
X = transpose(data_std)  

wcss = []  

for k in 1:12
    result = kmeans(X', k)  
    push!(wcss, result.totalcost)
end

plot(1:12, wcss, marker=:circle, xlabel="k")

k_group = 10
result = kmeans(X', k_group) 

labels = result.assignments 

println(labels[1:10]) 



n_plot = 3 
n_time = size(data, 1)


Random.seed!(123)

plot_layout = @layout [a{0.1h}; b{0.1h}; c{0.1h}; d{0.1h}; e{0.1h}; f{0.1h}; g{0.1h}; h{0.1h}; i{0.1h}; j{0.1h}]
p = plot(layout = plot_layout, size=(800, 2000))

for cluster_id in 1:k_group

    case_indices = findall(labels .== cluster_id)

    sample_indices = length(case_indices) ≤ n_plot ? case_indices :
                     sample(case_indices, n_plot; replace=false)

    for i in sample_indices
        plot!(p[cluster_id], 1:n_time, data[:, i], label="case $i")
    end

    title!(p[cluster_id], "group $cluster_id")
    xlabel!(p[cluster_id], "time")
    ylabel!(p[cluster_id], "conc")
end

display(p)








model_params_test = ref_params_test
sza = model_params_test[4:4,:,:]
model_params_test = cat(model_params_test[2:2,:,:],model_params_test[5:5,:,:], model_params_test[4:4,:,:]; dims=1)
@parameters sza press tempk1 tempk2
@parameters emit[1:n_latent_emit_species]
@parameters k[1:size(model_params_test)[1]]
k = Num[
    sza;
    press;
    tempk1;
    tempk2;
    ]
k_params_test = cat(max.(cos.(model_params_test[3:3,:,:]), 0),
               model_params_test[2:2,:,:], 
               exp.(model_params_test[1:1,:,:].^-1),
               exp.((model_params_test[1:1,:,:].^-1).*(-1)),
               ;dims=1
              )


@variables t
@species u(t)[1:n_latent_species] [description="State variables for simulation"]
@variables x[1:n_latent_species] [description="State variables for optimization"]
@variables dxdt[1:n_latent_species] [description="Taget derivative"]

function calc_basis_size(u)
    #	umat = zeros(T, nrxn, length(u)+1)
    #	rxs = []
    #	iξ = 1
        irx = 1
        for i in eachindex(u)
            # 0-order reactions
            ##  Ø -> A 
    #		rc = sum([abs.(ξ[iξ+j-1])*k[j] for j in 1:nrate])
    #		push!(rxs, Reaction(rc, nothing, [u[i]], nothing, [1]))
    #		umat[irx, 1] = 1
    #		iξ += nrate
    #		irx += 1
    
            # 1-st order reactions
            ## A -> Ø
    #		rc = sum([abs.(ξ[iξ+j-1])*k[j] for j in 1:nrate])
    #		push!(rxs, Reaction(rc, [u[i]], nothing, [1], nothing))
    #		umat[irx,i+CartesianIndex(1)] = 1
    #		iξ += nrate
    		irx += 1
            
            ## A -> B   30
            for j in eachindex(u)
                if i==j
                    continue
                end
    #			rc = sum([abs.(ξ[iξ+jj-1])*k[jj] for jj in 1:nrate])
    #			push!(rxs, Reaction(rc, [u[i]], [u[j]], [1], [1]))
    #			umat[irx,i+CartesianIndex(1)] = 1
    #			iξ += nrate
                irx += 1
            end
    
            # 2-nd order reactions
            ## 2A -> B   30
            for j in eachindex(u)
                if i==j continue end
    #            rc = sum([abs.(ξ[iξ+jj-1])*k[jj] for jj in 1:nrate])
    #            push!(rxs, Reaction(rc, [u[i]], [u[j]], [2], [1]))
    #            umat[irx,i+CartesianIndex(1)] = 2
    #            iξ += nrate
                irx += 1
            end
    
            ## A + B -> C   60
            for j in eachindex(u)
                for h in eachindex(u)
                    if (i == j) || (i == h) || (j == h) || i >= j continue end
    #                rc = sum([abs.(ξ[iξ+jj-1])*k[jj] for jj in 1:nrate])
    #                push!(rxs, Reaction(rc, [u[i],u[j]], [u[h]], [1,1], [1]))
    #                umat[irx,i+CartesianIndex(1)] = 1
    #                umat[irx,j+CartesianIndex(1)] = 1
    #                iξ += nrate
                    irx += 1
                end
            end
            # 3-rd order reactions
            ## 3A -> B
            for j in eachindex(u)
                if i==j continue end
     #           rc = sum([abs.(ξ[iξ+jj-1])*k[jj] for jj in 1:nrate])
     #           push!(rxs, Reaction(rc, [u[i]], [u[j]], [3], [1]))
    #            umat[irx,i+CartesianIndex(1)] = 3
     #           iξ += nrate
                irx += 1
            end
            
            ## 2A + B -> C ##120
            for j in eachindex(u)
                for h in eachindex(u)
                    if (i == j) || (i == h) || (j == h)   continue end
    #                rc = sum([abs.(ξ[iξ+jj-1])*k[jj] for jj in 1:nrate])
     #               push!(rxs, Reaction(rc, [u[i],u[j]], [u[h]], [2,1], [1]))
     #               umat[irx,i+CartesianIndex(1)] = 2
     #               umat[irx,j+CartesianIndex(1)] = 1
    #                iξ += nrate
                    irx += 1
                end
            end
    
            ## A + B + C -> D  60
            for j in eachindex(u)
                if i==j continue end
                for h in eachindex(u)
                    for g in eachindex(u)
                        if (i == j) || (i == h) || (i == g) || (j == h) || (j == g) || (h == g) || i > j || i > h || j > h continue end
    #                    rc = sum([abs.(ξ[iξ+jj-1])*k[jj] for jj in 1:nrate])
    #                    push!(rxs, Reaction(rc, [u[i],u[j],u[h]], [u[g]], [1,1,1], [1]))
    #                    umat[irx,i+CartesianIndex(1)] = 1
    #                    umat[irx,j+CartesianIndex(1)] = 1
    #                    umat[irx,h+CartesianIndex(1)] = 1
    #                    iξ += nrate
                        irx += 1
                    end
                end
            end   

    
    
            
            
    
        end
    #	rxs, umat
        irx-1
    end
    nrxn = calc_basis_size(u)

    n_latent_rxn_constant = length(k)
    simady_basis_size = nrxn * n_latent_rxn_constant
    basis_size = simady_basis_size

    ## sparse coefficient ξ
    @parameters ξ[1:basis_size]


    function create_basis(T, u, ξ, basis_size, n_rxn, nrate)

        umat = zeros(T, nrxn, length(u)+1)
        rxs = []
        iξ = 1
        irx = 1
        for i in eachindex(u)
            # 0-order reactions
            ##  Ø -> A 
    		#rc = sum([abs.(ξ[iξ+j-1])*k[j] for j in 1:nrate])
    		#push!(rxs, Reaction(rc, nothing, [u[i]], nothing, [1]))
    		#umat[irx, 1] = 1
    		#iξ += nrate
    		#irx += 1
    
            # 1-st order reactions
            ## A -> Ø
    		rc = sum([abs.(ξ[iξ+j-1])*k[j] for j in 1:nrate])
    		push!(rxs, Reaction(rc, [u[i]], nothing, [1], nothing))
    		umat[irx,i+CartesianIndex(1)] = 1
    		iξ += nrate
    		irx += 1
            
            ## A -> B   30
            for j in eachindex(u)
                if i==j
                    continue
                end
    			rc = sum([abs.(ξ[iξ+jj-1])*k[jj] for jj in 1:nrate])
    			push!(rxs, Reaction(rc, [u[i]], [u[j]], [1], [1]))
    			umat[irx,i+CartesianIndex(1)] = 1
    			iξ += nrate
                irx += 1
            end
    
            # 2-nd order reactions
            ## 2A -> B   30
            for j in eachindex(u)
                if i==j continue end
                rc = sum([abs.(ξ[iξ+jj-1])*k[jj] for jj in 1:nrate])
                push!(rxs, Reaction(rc, [u[i]], [u[j]], [2], [1]))
                umat[irx,i+CartesianIndex(1)] = 2
                iξ += nrate
                irx += 1
            end
    
            ## A + B -> C   60
            for j in eachindex(u)
                for h in eachindex(u)
                    if (i == j) || (i == h) || (j == h) || i > j continue end
                    rc = sum([abs.(ξ[iξ+jj-1])*k[jj] for jj in 1:nrate])
                    push!(rxs, Reaction(rc, [u[i],u[j]], [u[h]], [1,1], [1]))
                    umat[irx,i+CartesianIndex(1)] = 1
                    umat[irx,j+CartesianIndex(1)] = 1
                    iξ += nrate
                    irx += 1
                end
            end
            # 3-rd order reactions
            ## 3A -> B
            for j in eachindex(u)
                if i==j continue end
                rc = sum([abs.(ξ[iξ+jj-1])*k[jj] for jj in 1:nrate])
                push!(rxs, Reaction(rc, [u[i]], [u[j]], [3], [1]))
                umat[irx,i+CartesianIndex(1)] = 3
                iξ += nrate
                irx += 1
            end
            
            ## 2A + B -> C ##120
            for j in eachindex(u)
                for h in eachindex(u)
                    if (i == j) || (i == h) || (j == h)   continue end
                    rc = sum([abs.(ξ[iξ+jj-1])*k[jj] for jj in 1:nrate])
                    push!(rxs, Reaction(rc, [u[i],u[j]], [u[h]], [2,1], [1]))
                    umat[irx,i+CartesianIndex(1)] = 2
                    umat[irx,j+CartesianIndex(1)] = 1
                   iξ += nrate
                    irx += 1
                end
            end
    
            ## A + B + C -> D  60
            for j in eachindex(u)
                if i==j continue end
                for h in eachindex(u)
                    for g in eachindex(u)
                        if (i == j) || (i == h) || (i == g) || (j == h) || (j == g) || (h == g) || i > j || i > h || j > h continue end
                        rc = sum([abs.(ξ[iξ+jj-1])*k[jj] for jj in 1:nrate])
                        push!(rxs, Reaction(rc, [u[i],u[j],u[h]], [u[g]], [1,1,1], [1]))
                       umat[irx,i+CartesianIndex(1)] = 1
                        umat[irx,j+CartesianIndex(1)] = 1
                        umat[irx,h+CartesianIndex(1)] = 1
                        iξ += nrate
                        irx += 1
                    end
                end
            end   

    
    
    
    
            
    
        end
        rxs, umat
    end


basis, umat = create_basis(Float64, u, ξ, simady_basis_size, nrxn, n_latent_rxn_constant)
rsys = ReactionSystem(basis; name=:simady)
stoich = netstoichmat(rsys) 
basis

JLD2.@load "../model/simady/stage_5_n_latent_species_$(n_latent_species).jld"  ps_cpu

ps = (ps_cpu)
encoder, sparse_coeff = ps[1:size_encoder], ps[size_encoder+1:end]
encoder = reshape(encoder,(n_latent_species,:))


rates_ = []
for i in 0:size(basis)[1]-1
   
basis_ = substitute(basis[i+1].rate, Dict(ξ[i*4+1]=> (sparse_coeff)[i*4+1]))
basis_ = substitute(basis_, Dict(ξ[i*4+2]=> (sparse_coeff)[i*4+2]))
basis_ = substitute(basis_, Dict(ξ[i*4+3]=> (sparse_coeff)[i*4+3]))
basis_ = substitute(basis_, Dict(ξ[i*4+4]=> (sparse_coeff)[i*4+4]))
    push!(rates_, basis_)
end
rates_

filtered_basis = []
for i in 1-1:length(rates_)-1
    if rates_[i+1] !== 0f0
        temp_basis = basis[i+1]
        basis_ = substitute(temp_basis.rate, Dict(ξ[i*4+1]=> (sparse_coeff)[i*4+1]))
        basis_ = substitute(basis_, Dict(ξ[i*4+2]=> (sparse_coeff)[i*4+2]))
        basis_ = substitute(basis_, Dict(ξ[i*4+3]=> (sparse_coeff)[i*4+3]))
        basis_ = substitute(basis_, Dict(ξ[i*4+4]=> (sparse_coeff)[i*4+4]))
        temp_basis = @set temp_basis.rate = basis_
        push!(filtered_basis, temp_basis)
        
    end
end
filtered_basis

reaction_index = []
for i in 1:size(rates_)[1]
    if rates_[i] !== 0.0f0
        push!(reaction_index, i)
    end
end

sparse_coeff_filtered = []
for i in 1:size(sparse_coeff)[1]÷4
    if sparse_coeff[(i-1)*4+1] !== 0f0 || sparse_coeff[(i-1)*4+2] !== 0f0 || sparse_coeff[(i-1)*4+3] !== 0f0 || sparse_coeff[(i-1)*4+4] !== 0f0
        push!(sparse_coeff_filtered, sparse_coeff[(i-1)*4+1], sparse_coeff[(i-1)*4+2], sparse_coeff[(i-1)*4+3], sparse_coeff[(i-1)*4+4])
    end
end
sparse_coeff_filtered = Float64.(sparse_coeff_filtered)

umat_filtered = umat[reaction_index,:]
basis_filtered = basis[reaction_index]
stoich_filtered = stoich[:,reaction_index]

umat_filtered_gpu = cu(umat_filtered)
stoich_filtered_gpu = cu(stoich_filtered)





function oderatelaws(ξ, k, oneplusu, umat)
	ξabsmasked = abs.(ξ)
	ξmat = transpose(reshape(ξabsmasked, n_latent_rxn_constant, :))
    umat = reshape(umat, size(umat)[1], size(umat)[2], 1)
    oneplusu = reshape(oneplusu, 1, size(oneplusu)[1], size(oneplusu)[2])
    (ξmat * k) .* reshape(prod(oneplusu.^umat; dims=2),(size(umat)[1],:))
     
end


function dudt(ps, k, u, stoich, umat, weight)
    u = reshape(u, (size(u)[1],:))
    k = reshape(k, (size(k)[1],:))
	oneplusu = [1; u]
	ratelaws = oderatelaws(ps, k, oneplusu, umat)
	(stoich * ratelaws)
end



ref_data_test = abs.(ref_data_test)
ref_data_test = (ref_data_test .- ref_data_min) ./ (ref_data_max .- ref_data_min)
replace!(ref_data_test, NaN => 0.0)
ref_emit_test = ref_emit_test#
ref_emit_test_encoded = encoder_(encoder, (ref_emit_test), nruns)
ref_data_test_encoded = encoder_(encoder, (ref_data_test), nruns)

# function run_c_test(ps, dataset, dc_std) 
#     sparse_coeff = cu(ps)
#     c, e, P = dataset
#      #|> cpu#.*  latent_species_dcdt_std  .+ e[:,:,i_case].*60)
#      e = cu(e ./ dc_std )
#      dc_std = cu(dc_std)
#     P = cu(P)

#     ref_data_encoded_pred = ones(n_latent_species, ntimestep, nruns)

    
#     for i_case in 1:nruns
#         println(i_case)
#         dcdt_pred(u) = (dudt((sparse_coeff), P[:,:,i_case], (u), (stoich_filtered_gpu), (umat_filtered_gpu), 1.0))
#         function sindy_ude_pred!(du, u, p ,t)
#             du .= (dcdt_pred(u)[:,Int(t÷60+1)]    .+ e[:,Int(t÷60+1), i_case]) .*  dc_std[:]
#         end
#         prob2 = ODEProblem(sindy_ude_pred!, cu(c[:, 1, i_case]), (times[1], times[end]))
#         sol_sindy_ude = cpu(Array(solve(prob2, Tsit5(), saveat=60)))
#         ref_data_encoded_pred[:,:,i_case] .= sol_sindy_ude
#     end
#     return ref_data_encoded_pred

# end
# ref_data_encoded_test_pred = run_c_test(cu(sparse_coeff_filtered), [(ref_data_test_encoded), (ref_emit_test_encoded), (k_params_test)], dc_std)

# ref_data_test_pred = decoder_(cpu(encoder), (ref_data_encoded_test_pred), nruns)
# ref_data_test_pred .= ref_data_test_pred .*(ref_data_max.-ref_data_min) .+ ref_data_min 
ref_data_test = ref_data_test .* (ref_data_max.-ref_data_min) .+ ref_data_min

i_spec = 10
# JLD2.jldsave( "ref_data_test_pred.jld"; ref_data_test_pred)
JLD2.@load "../ref_data_test_pred.jld" ref_data_test_pred
# open("plots/simady/rmse_test.txt", "w") do file
#     write(file, "$(specname[i_spec]); rmse = $(rmsd(ref_data_test_pred[i_spec,:,:],ref_data_test[i_spec,:,:])), rms = $(mean(ref_data_test[i_spec,:,:].^2)^0.5) \n")
#     write(file, "overall; rmse = $(rmsd(ref_data_test_pred[:,:,:],ref_data_test[:,:,:])), rms = $(mean(ref_data_test[:,:,:].^2)^0.5)) \n")
# end


percent = 100
err_ozone = ref_data_test_pred[i_spec,:,:] - ref_data_test[i_spec,:,:]
upper_ozone_100, lower_ozone_100 = [], []
for i in 1:ntimestep
    push!(upper_ozone_100, percentile(err_ozone[i,:],percent+(100-percent)/2))
    push!(lower_ozone_100, percentile(err_ozone[i,:],(100-percent)/2))
end
percent = 90
upper_ozone_90, lower_ozone_90 = [], []
for i in 1:ntimestep
    push!(upper_ozone_90, percentile(err_ozone[i,:],percent+(100-percent)/2))
    push!(lower_ozone_90, percentile(err_ozone[i,:],(100-percent)/2))
end
percent = 80
upper_ozone_80, lower_ozone_80 = [], []
for i in 1:ntimestep
    push!(upper_ozone_80, percentile(err_ozone[i,:],percent+(100-percent)/2))
    push!(lower_ozone_80, percentile(err_ozone[i,:],(100-percent)/2))
end
import ColorSchemes.grayC100
plot_error =  plot(times./1440,mean(err_ozone;dims=2)[:]; 
                   ribbon = (-lower_ozone_100, upper_ozone_100 ),
                   color = grayC100[75],
                   label="100%", 
                   xlabel="Time (day)",
                   xticks=(0:2:10),
                   titlefontsize = 16,
                   xtickfontsize = 16, ytickfontsize = 16, 
                   xlabelfontsize = 18, ylabelfontsize = 18,
                   legendfontsize = 20,
                   linewidth=0,
                   left_margin = 8Plots.mm,
                   bottom_margin = 12Plots.mm,
                   formatter = identity,
                   size=(1200, 300)
                   )
plot!(times./1440,mean(err_ozone;dims=2);linewidth=0, ribbon = (-lower_ozone_90, upper_ozone_90 ), color = grayC100[50], label="90%")
plot!(times./1440,mean(err_ozone;dims=2); linewidth=0, ribbon = (-lower_ozone_80, upper_ozone_80 ), color = grayC100[25], label="80%")
labels
rmsebycase = []
for i in 1:nruns
    push!(rmsebycase, rmsd(ref_data_test_pred[i_spec,:,:][:,i],ref_data_test[i_spec,:,:][:,i]))
end
rmsebycase
histogram(labels)


begin


k = maximum(labels)
n_case = length(labels)
n_time = size(ref_data_test, 2)


plot_grid = @layout [a{0.1h}; b{0.1h}; c{0.1h}; d{0.1h}; e{0.1h}; f{0.1h}; g{0.1h}; h{0.1h}; i{0.1h}; j{0.1h}]
p_k_mean = plot(layout = (k, 3), size=(1500, 170 * k))

plot_index = 1

for group_id in 1:k

    group_indices = findall(labels .== group_id)

    if length(group_indices) < 3
        @warn "Group $group_id has less than 3 samples, skipping..."
        continue
    end

    sorted_idx = sort(group_indices, by = i -> rmsebycase[i])
    case_ids = [
        sorted_idx[1], 
        sorted_idx[cld(length(sorted_idx), 2)],
        sorted_idx[end] 
    ]

    for (j, case_id) in enumerate(case_ids)
        plot!(p_k_mean[plot_index], 
              1:n_time, 
              ref_data_test[10,:, case_id], 
              label=:none, 
              lw=2,
              xlabelfontsize = 23,
              ylabelfontsize = 23,
              xticks = (0:24:n_time, collect(0:length(0:24:n_time)-1)),
              xtickfontsize = 12,
              ytickfontsize = 12,
              titlefontsize = 23,
              left_margin = (j % 3 == 1 ? 12Plots.mm : (j % 3 == 2 ? 2Plots.mm : 4Plots.mm)),
              right_margin = (j % 3 == 1 ? 3Plots.mm : (j % 3 == 2 ? -1Plots.mm : -1Plots.mm)),
              color = :black,
              )
        plot!(p_k_mean[plot_index], 
              1:n_time, 
              ref_data_test_pred[10,:, case_id], 
              label=:none, 
              lw=2, 
              ls=:dash,
              xlabelfontsize = 33,
              ylabelfontsize = 23,
              xtickfontsize = 12,
              ytickfontsize = 12,
              titlefontsize = 33,
              xticks = (0:24:n_time, collect(0:length(0:24:n_time)-1)),
              left_margin = (j % 3 == 1 ? 12Plots.mm : (j % 3 == 2 ? 2Plots.mm : 4Plots.mm)),
              right_margin = (j % 3 == 1 ? 3Plots.mm : (j % 3 == 2 ? -1Plots.mm : -1Plots.mm)),
              color = :red,
              )
        title!(p_k_mean[plot_index], (j % 3 == 1 && group_id == 1 ? "Best" : (j % 3 == 2 && group_id == 1 ? "Median" : (j%3 == 0 && group_id == 1 ? "Worst" : "" ))))
        xlabel!(p_k_mean[plot_index], (j%3 == 2 && group_id == 10 ? "Time (day)" : "" ))
        ylabel!(p_k_mean[plot_index], (j%3 == 1 ? string(group_id) : "" ))
        plot_index += 1
    end
end

display(p_k_mean)

end

savefig(p_k_mean, "k_mean_case.svg")

