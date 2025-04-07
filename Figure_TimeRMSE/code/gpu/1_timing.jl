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
    using BenchmarkTools
    using DifferentialEquations
end


using CUDA
CUDA.allowscalar(false)
CUDA.memory_status() 

n_latent_species = 6

n_sim = 1000
ndays = 10

saveat = 60.0
dt = 60 # minutes
nruns = 1070
n_species = 5838
nmete = 5
ntimestep = 24*ndays +1
times = LinRange(0, saveat * (ndays * 24), ntimestep)
startspec = 1


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

JLD2.@load "c_test_utils.jld" ref_data_max ref_data_min dc_std
JLD2.@load "testing_set.jld" ref_data_test ref_emit_test ref_params_test specname

model_params_test = ref_params_test
sza = model_params_test[4:4,:,:]
model_params_test = cat(model_params_test[2:2,:,:],model_params_test[5:5,:,:], model_params_test[4:4,:,:]; dims=1)

# Sympolics
## mete and emit
@parameters sza press tempk
@parameters emit[1:n_latent_species]
@parameters k[1:size(model_params_test)[1]]
k = Num[
    sza;
    press;
    tempk;
    ]

k_params_test = cat(max.(cos.(model_params_test[3:3,:,:]), 0),
               model_params_test[2:2,:,:], 
               exp.(model_params_test[1:1,:,:].^-1),
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
#basis = basis 
umat = umat #|> gpu
rsys = ReactionSystem(basis; name=:simady)
stoich = netstoichmat(rsys) #|> gpu
basis
JLD2.@load "epoch_5000.jld" ps_cpu
ps = ps_cpu
encoder, sparse_coeff = ps[1:size_encoder], ps[size_encoder+1:end]
encoder = reshape(encoder,(n_latent_species,:))

rates_ = []
for i in 0:size(basis)[1]-1
   
basis_ = substitute(basis[i+1].rate, Dict(ξ[i*3+1]=> (sparse_coeff)[i*3+1]))
basis_ = substitute(basis_, Dict(ξ[i*3+2]=> (sparse_coeff)[i*3+2]))
basis_ = substitute(basis_, Dict(ξ[i*3+3]=> (sparse_coeff)[i*3+3]))
    push!(rates_, basis_)
end
rates_

filtered_basis = []
for i in 1-1:length(rates_)-1
    if rates_[i+1] !== 0f0
        temp_basis = basis[i+1]
        basis_ = substitute(temp_basis.rate, Dict(ξ[i*3+1]=> (sparse_coeff)[i*3+1]))
        basis_ = substitute(basis_, Dict(ξ[i*3+2]=> (sparse_coeff)[i*3+2]))
        basis_ = substitute(basis_, Dict(ξ[i*3+3]=> (sparse_coeff)[i*3+3]))
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
reaction_index = []
for i in 1:size(rates_)[1]
    if rates_[i] !== 0.0f0
        push!(reaction_index, i)
    end
end
sparse_coeff_filtered = []
for i in 1:size(sparse_coeff)[1]÷3
    if sparse_coeff[(i-1)*3+1] !== 0f0 || sparse_coeff[(i-1)*3+2] !== 0f0 || sparse_coeff[(i-1)*3+3] !== 0f0
        push!(sparse_coeff_filtered, sparse_coeff[(i-1)*3+1], sparse_coeff[(i-1)*3+2], sparse_coeff[(i-1)*3+3])
    end
end
sparse_coeff_filtered = Float64.(sparse_coeff_filtered)

umat_filtered = umat[reaction_index,:]
basis_filtered = basis[reaction_index]
stoich_filtered = stoich[:,reaction_index]

umat_filtered_gpu = cu(umat_filtered)
stoich_filtered_gpu = cu(stoich_filtered)




function oderatelaws(ξ, k, oneplusu, umat, i_case)
    n_case = size(i_case)[1]
    ξabsmasked = abs.(ξ)
    ξmat = transpose(reshape(ξabsmasked, n_latent_rxn_constant, :))
    umat = reshape(umat, size(umat)[1], size(umat)[2], 1)
    oneplusu = reshape(oneplusu, 1, size(oneplusu)[1], size(oneplusu)[2])
    reshape(ξmat * k, size(ξmat)[1], size(times)[1], n_case) .* prod(oneplusu.^umat; dims=2)
     
end


function dudt(ps, k, u, stoich, umat, weight, i_case)

    u = reshape(u, (size(u)[1],:))
    k = reshape(k, (size(k)[1],:))
    oneplusu = [1; u]
    ratelaws = oderatelaws(ps, k, oneplusu, umat, i_case)
    ratelaws = reshape(ratelaws, size(stoich)[2], :)
    reshape((stoich * ratelaws), n_latent_species, size(times)[1], size(i_case)[1])
end

ref_data_test = abs.(ref_data_test)
ref_data_test = (ref_data_test .- ref_data_min) ./ (ref_data_max .- ref_data_min)
replace!(ref_data_test, NaN => 0.0)
ref_emit_test = ref_emit_test#
ref_emit_test_encoded = encoder_(encoder, (ref_emit_test), nruns)
ref_data_test_encoded = encoder_(encoder, (ref_data_test), nruns)



    println("run test case, surrogate model, GPU:")
    sparse_coeff_filtered_gpu = cu(sparse_coeff_filtered)
    e = cu(ref_emit_test_encoded) ./ cu(dc_std)
    dc_std = cu(dc_std)

    c_gpu_part = cu(cat([ref_data_test_encoded[:, :, 1:n_sim] for _ in 1:n_sim]...;dims=3))
    e_gpu_part = cat([e[:, :, 1:n_sim] for _ in 1:n_sim]...;dims=3)
    P_gpu_part =  cu(cat([(k_params_test[:, :, 1:n_sim]) for _ in 1:10n_sim00]...;dims=3))

    dcdt_pred(u,i) = (dudt((sparse_coeff_filtered_gpu), P_gpu_part[:, :, i], (u), (stoich_filtered_gpu), (umat_filtered_gpu), 1.0, i))
    function sindy_ude_pred!(du, u, p, t)
        i = p
        du .= (dcdt_pred(u,i)[:, Int(t ÷ 60 + 1),:] .+ e_gpu_part[:, Int(t ÷ 60 + 1), i]) .* dc_std[:]
    end

#begin
   
    i_case = 1:n_sim
    prob2 = ODEProblem(sindy_ude_pred!, c_gpu_part[:, 1, i_case],  (times[1], times[end]))
    bench_gpu = @benchmark sol_sindy_ude = solve(prob2, Tsit5(), p = i_case, saveat=60)
    GC.gc()


JLD2.jldsave("timing/bench_multiple_$(n_sim).jld"; bench_gpu)