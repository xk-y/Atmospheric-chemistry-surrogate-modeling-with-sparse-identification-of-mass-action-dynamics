


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
    pwd()
#################### Part 1: generate reference data ####################
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
# initialize an encoder, create functions of encoding/decoding processes
encoder = abs.(Flux.glorot_uniform(Random.seed!(seed), n_latent_species, n_species)) 
size_encoder = size(reshape(encoder,:))[1]
#encoder = ones(n_latent_species, n_species) .* 1e-3 
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
#n_latent_species = 10
n_latent_emit_species = n_latent_species


JLD2.@load "c_test_utils.jld" ref_data_max ref_data_min dc_std
JLD2.@load "../../testing_set.jld" ref_data_test ref_emit_test ref_params_test specname




model_params_test = ref_params_test#ref_params
sza = model_params_test[4:4,:,:]
model_params_test = cat(model_params_test[2:2,:,:],model_params_test[5:5,:,:], model_params_test[4:4,:,:]; dims=1)
# Sympolics
## mete and emit
@parameters sza press tempk1 tempk2
@parameters emit[1:n_latent_emit_species]
#k = Num[sza; press; tempk/100.0; emit]
@parameters k[1:size(model_params_test)[1]]
#number of latent reaction rate Constant
k = Num[
    sza;
    
    #sza1;
    #sza2;
    #cos(sza);
    #sin(sza.*2.0);
    #cos(sza.*2.0);
    press;
    #(tempk^2)*exp(tempk^-1);
    tempk1;
    tempk2;
    #emit;
    #cos(sza);
    #sin(sza);
    ]
k_params_test = cat(max.(cos.(model_params_test[3:3,:,:]), 0),
               model_params_test[2:2,:,:], 
               #(model_params[1:1,:,:].^2).*exp.(model_params[1:1,:,:].^-1),
               exp.(model_params_test[1:1,:,:].^-1),
               exp.((model_params_test[1:1,:,:].^-1).*(-1)),
               #model_params[4:end,:,:]
               #cos.(model_params[4:4,:,:])
               #sin.(model_params[4:4,:,:])
               ;dims=1
              )



## state variable
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
    
    #emis_basis_size = n_latent_emit_species #* n_latent_emit_species
    basis_size = simady_basis_size #+ emis_basis_size
    ## sparse coefficient ξ
    
    @parameters ξ[1:basis_size] #[bounds=(0.0, Inf)]


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
#basis = basis 
umat = umat #|> gpu
rsys = ReactionSystem(basis; name=:simady)
stoich = netstoichmat(rsys) #|> gpu
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
#basis_ = substitute(basis_, Dict(sza => mean(k_params[1,:,1])))
#basis_ = substitute(basis_, Dict(press => mean(k_params[2,:,1])))
#basis_ = substitute(basis_, Dict(tempk => mean(k_params[3,:,1])))
    push!(rates_, basis_)
end
rates_

filtered_basis = []
for i in 1-1:length(rates_)-1
    if rates_[i+1] !== 0f0
        temp_basis = basis[i+1]
        #temp_basis = @set temp_basis.rate = 1.0
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
    #ξ_simady = ps[1:simady_basis_size]
    #ξ_emis = ps[simady_basis_size+1:simady_basis_size+emis_basis_size]
    #ξ, ps_nn = ps[1:basis_size], ps[basis_size+1:end]
    #nn = re(ps_nn)
    u = reshape(u, (size(u)[1],:))
    k = reshape(k, (size(k)[1],:))
    #k_emit = reshape(k_emit, (size(k_emit)[1],:))
	oneplusu = [1; u]
	ratelaws = oderatelaws(ps, k, oneplusu, umat)# for (col_k, col_u) in (eachcol(k), eachcol(oneplusu))]...;dims=2)
    #ξ_emis_mat = abs.(ξ_emis)#transpose(reshape(ξ_emis, n_latent_emit_species, n_latent_emit_species))
    #println(size(ξ_emis_mat))
    #println(size(k_emit))
	(stoich * ratelaws) #.+ (ξ_emis_mat .* k_emit)
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


# JLD2.jldsave( "ref_data_test_pred.jld"; ref_data_test_pred)
JLD2.@load "../ref_data_test_pred.jld" ref_data_test_pred
# open("plots/simady/rmse_test.txt", "w") do file
#     write(file, "$(specname[i_spec]); rmse = $(rmsd(ref_data_test_pred[i_spec,:,:],ref_data_test[i_spec,:,:])), rms = $(mean(ref_data_test[i_spec,:,:].^2)^0.5) \n")
#     write(file, "overall; rmse = $(rmsd(ref_data_test_pred[:,:,:],ref_data_test[:,:,:])), rms = $(mean(ref_data_test[:,:,:].^2)^0.5)) \n")
# end
# air pollutants (NOx, SOx, CO, O3, etc.) and greenhouse gasses (CO2, CH4, N2O, etc.)
idx_concerned = []
for spec in ["O3","NO","NO2","NO3","HO2","OH"]#["T123CATECH", "OTNCATECO", "C32OH13CO", "CO2H3CO3H", "BCALAO", "OETOLO", "NC61CO3H", "C3MCOCO2H", "C6DICARB", "MMCFO", "OXYBIPENO3", "INDOOH", "BCALCO", "HO1C3OOH", "DNMETOL", "CH3OOH", "CH3OH"]#["O3","NO","NO2","NO3","CH3OH"]
    idx = findfirst(x -> x == spec, specname)
    push!(idx_concerned,idx)
end
idx_concerned
i_spec = idx_concerned[2]
ref_data_test_pred[i_spec,:,:]

for i_spec in idx_concerned#[5:5]
percent = 100
err_ozone = ref_data_test_pred[i_spec,:,:] - ref_data_test[i_spec,:,:]
upper_ozone_100, lower_ozone_100 = [], []
for i in 1:ntimestep#length(times_test)
    push!(upper_ozone_100, percentile(err_ozone[i,:],percent+(100-percent)/2))
    push!(lower_ozone_100, percentile(err_ozone[i,:],(100-percent)/2))
end
percent = 90
#err_ozone_100 = ref_data_pred[i_spec,:,:] - ref_data_train[i_spec,:,:]
upper_ozone_90, lower_ozone_90 = [], []
for i in 1:ntimestep#length(times_test)
    push!(upper_ozone_90, percentile(err_ozone[i,:],percent+(100-percent)/2))
    push!(lower_ozone_90, percentile(err_ozone[i,:],(100-percent)/2))
end
percent = 80
#err_ozone_80 = ref_data_pred[i_spec,:,:] - ref_data_train[i_spec,:,:]
upper_ozone_80, lower_ozone_80 = [], []
for i in 1:ntimestep#length(times_test)
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
                   ylabel = "$(specname[i_spec]) Error(ppb)",
                   titlefontsize = 16,
                   xtickfontsize = 16, ytickfontsize = 16, 
                   xlabelfontsize = 18, ylabelfontsize = 18,
                   legendfontsize = 20,
                   linewidth=0,
                   #yticks=[-0.3, -0.2,-0.1,0,0.1,0.2,0.3],
                   left_margin = 8Plots.mm,
                   bottom_margin = 12Plots.mm,
                   formatter = identity,
                   size=(1200, 300)
                   )
plot!(times./1440,mean(err_ozone;dims=2);linewidth=0, ribbon = (-lower_ozone_90, upper_ozone_90 ), color = grayC100[50], label="90%")
plot!(times./1440,mean(err_ozone;dims=2); linewidth=0, ribbon = (-lower_ozone_80, upper_ozone_80 ), color = grayC100[25], label="80%")

rmsebycase = []
for i in 1:nruns
    push!(rmsebycase, rmsd(ref_data_test_pred[i_spec,:,:][:,i],ref_data_test[i_spec,:,:][:,i]))
end
rmsebycase




function plot_extreme_cases(scenario, err)

    function plot_case(ref, pred, pred_nobuff, time, ind, err_, scenario)
        p = [] 
        no = 1
        p_title = plot(title = "$(scenario)",titlefontsize = 20, grid = false, showaxis = false, bottom_margin = -160Plots.px)
        push!(p, p_title)
        for i in ind
            
            ptemp = plot(time./1440, ref[:,i];
                         legend=((no==1 && scenario=="Worst") ? :topright : :none), 
                         labels="Reference",
                         #title = "rmse = $(round(err_[i]; digits = 3))",
                         
                         xtickfontsize = 14, ytickfontsize = 14, 
                         xlabelfontsize = 18, ylabelfontsize = 18,
                         legendfontsize = 20,
                         xticks=(0:2:10),
                         #xlim=(0,4),
                         #xlabel = (scenario == "Median" && no == 3 ? "Time (day)" : ""),
                         ylabel = (scenario == "Best" && no == 2 ? "$(specname[i_spec]) (ppb)" : ""),
                         #ylim=((no==1 && scenario=="Worst") ? (0.0,0.335) : :best),
                         left_margin = (scenario == "Best" ? 8Plots.mm : 4Plots.mm ),
                         right_margin = (scenario == "Worst" ? 3Plots.mm : 1Plots.mm ),
                         bottom_margin = (no == 3 ? 7Plots.mm : 0Plots.mm ),
                         top_margin = (no == 1 ? 7Plots.mm : 3Plots.mm ),
                         color=:black,
                         formatter = identity)
            #plot!(time./1440, pred[:,i]; 
            #      labels="SINDy",
            #      linestyle=:dash, color=:red)
            plot!(time./1440, pred[:,i]; 
                  labels="SIMADy",
                  linestyle=:dash, color=:red)
            push!(p, ptemp)
            no+=1
        end
        plot(p...,size=(1200, 500),layout=(4,1))
    end

    if scenario == "Best"
        minicases = []
        local temp = err
        local inds = Array(1:length(err))
        for i in 1:3
                minicase = findmin(temp)[2]
                push!(minicases,inds[minicase])
                temp = temp[1:end .!= minicase]
                inds = inds[1:end .!= minicase]
        end
        println(minicases)
        plot_idx = setdiff(Array(1:length(err)), inds)
        return plotmin = plot_case(ref_data_test[i_spec,:,:], ref_data_test_pred[i_spec,:,:], 0, times, minicases, err, scenario),minicases
    end

    if scenario == "Worst"
        maxicases = []
        local temp = err
        local inds = Array(1:length(err))
        for i in 1:3
                maxicase = findmax(temp)[2]
                push!(maxicases,inds[maxicase])
                temp = temp[1:end .!= maxicase]
                inds = inds[1:end .!= maxicase]
        end
        println(maxicases)
        plot_idx = setdiff(Array(1:length(err)), inds)
        return plotmax = plot_case(ref_data_test[i_spec,:,:], ref_data_test_pred[i_spec,:,:], 0, times, maxicases, err, scenario),maxicases
    end

    if scenario == "Median"
        rmsebycase_index = hcat(err,1:length(err))
        rmsebycase_index_order = hcat(rmsebycase_index,sortperm(rmsebycase))
        sorted_rmsebycase_index_order = sortslices(rmsebycase_index_order,dims=1,by=x->x[1],rev=false)
        median_ind = [sorted_rmsebycase_index_order[Int(floor(length(rmsebycase)/2))-1,2],
        sorted_rmsebycase_index_order[Int(floor(length(rmsebycase)/2)),2],
        sorted_rmsebycase_index_order[Int(floor(length(rmsebycase)/2))+1,2]]
        println(median_ind)
        return plotmax = plot_case(ref_data_test[i_spec,:,:], ref_data_test_pred[i_spec,:,:], 0, times, median_ind, err, scenario),median_ind
    end



end
plot_case_best, best_case = plot_extreme_cases("Best",(rmsebycase))
plot_case_median, median_case = plot_extreme_cases("Median",(rmsebycase))
plot_case_worst, worst_case = plot_extreme_cases("Worst",(rmsebycase))
plot_case_bmw = plot(plot_case_best, plot_case_median, plot_case_worst, layout=(1,3))
savefig(plot_case_bmw, "plots/fig_cases_$(specname[i_spec]).svg")
savefig(plot_error, "plots/fig_percentile_$(specname[i_spec]).svg")

end
best_case_rmse = []
median_case_rmse = []
worst_case_rmse = []

for (i,index) in enumerate(best_case)
    rmse_temp = rmsd(ref_data_test_pred[i_spec,:,best_case[i]],ref_data_test[i_spec,:,best_case[i]])
    push!(best_case_rmse, rmse_temp)

end

for (i,index) in enumerate(median_case)
    rmse_temp = rmsd(ref_data_test_pred[i_spec,:,median_case[i]],ref_data_test[i_spec,:,median_case[i]])
    push!(median_case_rmse, rmse_temp)

end

for (i,index) in enumerate(worst_case)
    rmse_temp = rmsd(ref_data_test_pred[i_spec,:,worst_case[i]],ref_data_test[i_spec,:,worst_case[i]])
    push!(worst_case_rmse, rmse_temp)

end


nruns_tenpercent = Int(nruns*0.10)

function plot_extreme_cases_overunder_pred(scenario, err)

    function plot_case(ref, pred, pred_nobuff, time, ind, err_, scenario)
        p = [] 
        no = 1
        p_title = plot(title = "$(scenario)",titlefontsize = 20, grid = false, showaxis = false, bottom_margin = -160Plots.px)
        push!(p, p_title)
        for i in ind
            
            ptemp = plot(time./1440, ref[:,i];
                         legend=((no==1 && scenario=="Worst") ? :topright : :none), 
                         labels="Reference",
                         #title = "rmse = $(round(err_[i]; digits = 3))",
                         
                         xtickfontsize = 14, ytickfontsize = 14, 
                         xlabelfontsize = 18, ylabelfontsize = 18,
                         legendfontsize = 20,
                         xticks=(0:2:10),
                         #xlim=(0,4),
                         #xlabel = (scenario == "Median" && no == 3 ? "Time (day)" : ""),
                         ylabel = (scenario == "Best" && no == 2 ? "$(specname[i_spec]) (ppb)" : ""),
                         #ylim=((no==1 && scenario=="Worst") ? (0.0,0.335) : :best),
                         left_margin = (scenario == "Best" ? 8Plots.mm : 4Plots.mm ),
                         right_margin = (scenario == "Worst" ? 3Plots.mm : 1Plots.mm ),
                         bottom_margin = (no == 3 ? 7Plots.mm : 0Plots.mm ),
                         top_margin = (no == 1 ? 7Plots.mm : 3Plots.mm ),
                         color=:black,
                         formatter = identity)
            #plot!(time./1440, pred[:,i]; 
            #      labels="SINDy",
            #      linestyle=:dash, color=:red)
            plot!(time./1440, pred[:,i]; 
                  labels="SIMADy",
                  linestyle=:dash, color=:red)
            push!(p, ptemp)
            no+=1
        end
        plot(p...,size=(3000, 5000),layout=(32,:))
    end
    if scenario == "Worst"
        maxicases = []
        local temp = err
        local inds = Array(1:length(err))
        for i in 1:nruns_tenpercent
                maxicase = findmax(temp)[2]
                push!(maxicases,inds[maxicase])
                temp = temp[1:end .!= maxicase]
                inds = inds[1:end .!= maxicase]
        end
        println(maxicases)
        plot_idx = setdiff(Array(1:length(err)), inds)
        return plotmax = plot_case(ref_data_test[i_spec,:,:], ref_data_test_pred[i_spec,:,:], 0, times, maxicases, err, scenario),maxicases
    end

end

plot_case_worst_overunder_pred, worst_case_overunder_pred = plot_extreme_cases_overunder_pred("Worst",(rmsebycase))

begin
err_worst_case = []
count_underpred = 0
count_overpred = 0
for i in worst_case_overunder_pred
    err = (mean(ref_data_test_pred[i_spec,:,:][:,i]) .- mean(ref_data_test[i_spec,:,:][:,i]))
    if err > 0 
        count_overpred +=1
    elseif err < 0 
        count_underpred +=1
    end
    push!(err_worst_case, err)
end
end
count_underpred, count_overpred
err_worst_case





open("plots/simady/rmse_test_bycase.txt", "w") do file

        write(file, "$(specname[i_spec]); rmse_best = $(best_case_rmse), rms_best = $(mean(ref_data_test[i_spec,:,best_case[1]].^2)^0.5), $(mean(ref_data_test[i_spec,:,best_case[2]].^2)^0.5), $(mean(ref_data_test[i_spec,:,best_case[3]].^2)^0.5) \n")
        write(file, "$(specname[i_spec]); rmse_median = $(median_case_rmse), rms_median = $(mean(ref_data_test[i_spec,:,median_case[1]].^2)^0.5), $(mean(ref_data_test[i_spec,:,median_case[2]].^2)^0.5), $(mean(ref_data_test[i_spec,:,median_case[3]].^2)^0.5) \n")
        write(file, "$(specname[i_spec]); rmse_worst = $(worst_case_rmse), rms_worst = $(mean(ref_data_test[i_spec,:,worst_case[1]].^2)^0.5), $(mean(ref_data_test[i_spec,:,worst_case[2]].^2)^0.5), $(mean(ref_data_test[i_spec,:,worst_case[3]].^2)^0.5) \n")
        write(file, "count_underpred = $(count_underpred); count_overpred = $(count_overpred) \n")
    
end