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
    end


    using CUDA
    CUDA.allowscalar(false)
    CUDA.memory_status() 
    cd(Base.source_path()*"/..")
ndays = 4

saveat = 60.0
dt = 60 # minutes
nruns = 4350
n_species = 5838
nmete = 5
ntimestep = 24*ndays +1

startspec = 1

JLD2.@load "../training_set.jld" ref_data_train ref_emit_train ref_params_train specname
timelength = 60 * (ndays * 24) # minutes
dt = 60.0
startday = 2
times = LinRange(0, timelength, ntimestep)
tspan = (times[1], times[end])




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

model_params = ref_params_train
sza = model_params[4:4,:,:]
model_params = cat(model_params[2:2,:,:],model_params[5:5,:,:], model_params[4:4,:,:]; dims=1)

@parameters sza press tempk1 tempk2
@parameters emit[1:n_latent_emit_species]
@parameters k[1:size(model_params)[1]]
k = Num[
    sza;
    press;
    tempk1;
    tempk2;
    ]
k_params = cat(max.(cos.(model_params[3:3,:,:]), 0),
               model_params[2:2,:,:], 
               exp.(model_params[1:1,:,:].^-1),
               exp.((model_params[1:1,:,:].^-1).*(-1)),
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
umat = umat #|> gpu
rsys = ReactionSystem(basis; name=:simady)
stoich = netstoichmat(rsys) #|> gpu
basis

JLD2.@load "model/simady/stage_5_n_latent_species_$(n_latent_species).jld"  ps_cpu

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






ref_data_max = maximum(ref_data_train[:,:,:];dims=(2,3))
ref_data_min = minimum(ref_data_train[:,:,:];dims=(2,3))
ref_data_normalized = (ref_data_train[:,:,:] .- ref_data_min ) ./(ref_data_max.-ref_data_min)
replace!(ref_data_normalized, NaN => 0.0)
batchsize = 30
training_dataset = DataLoader((conc=(ref_data_normalized), emit=(ref_emit_train), p = (k_params)), batchsize = batchsize)

C0, E0, P0 = (first(training_dataset))

c = encoder_(cpu(encoder), ref_data_normalized, nruns)
dc = diff(c;dims=2)./dt
dc = cat(dc, dc[:,end:end,:];dims=2) #.- e.*60
dc_std = std(dc;dims=(2,3)) 
JLD2.jldsave( "c_test_utils.jld"; ref_data_max,ref_data_min,dc_std)

function run_c(ps, dataset, dc_std) 
    encoder, sparse_coeff = ps[1][1:size_encoder], ps[2]
    encoder = reshape(encoder,(n_latent_species,:))
    c, e, P = dataset
    c = encoder_(encoder,c,nruns)
    e = encoder_(encoder,e,nruns)
    e = e ./ dc_std |> gpu
    dc_std = cu(dc_std)
    sparse_coeff = sparse_coeff |> gpu
    ref_data_encoded_pred = ones(n_latent_species, ntimestep, nruns)
    
    for i_case in 1:nruns
        println(i_case)
        dcdt_pred(u) = dudt(sparse_coeff, cu(P[:,:,i_case]), (u), (stoich_filtered_gpu), (umat_filtered_gpu), 1.0)
        function sindy_ude_pred!(du, u, p ,t)
            du .= (dcdt_pred(u)[:,Int(t÷60+1)]  .+ e[:,Int(t÷60+1), i_case]) .*  (dc_std[:]) 
        end
        prob2 = ODEProblem(sindy_ude_pred!, cu(c)[:, 1, i_case], (times[1], times[end]))
        sol_sindy_ude = cpu(Array(solve(prob2, Tsit5(), saveat=60)))
        ref_data_encoded_pred[:,:,i_case] .= sol_sindy_ude
    end
    return ref_data_encoded_pred

end
ref_data_encoded_pred = run_c([ps, sparse_coeff_filtered], [training_dataset.data.conc,training_dataset.data.emit, training_dataset.data.p], (dc_std))

ref_data_pred = decoder_(cpu(encoder), (ref_data_encoded_pred), nruns)
ref_data_pred .= ref_data_pred .*(ref_data_max.-ref_data_min) .+ ref_data_min 
ref_data = cpu(training_dataset.data.conc) .* (ref_data_max.-ref_data_min) .+ ref_data_min

JLD2.jldsave( "ref_data_pred.jld"; ref_data_pred)
i_spec = 10

open("plots/simady/rmse_train.txt", "w") do file
    write(file, "$(specname[i_spec]); rmse = $(rmsd(ref_data_pred[i_spec,:,:],ref_data[i_spec,:,:])), rms = $(mean(ref_data[i_spec,:,:].^2)^0.5) \n")
    write(file, "overall; rmse = $(rmsd(ref_data_pred[:,:,:],ref_data[:,:,:])), rms = $(mean(ref_data[:,:,:].^2)^0.5) \n")
end

