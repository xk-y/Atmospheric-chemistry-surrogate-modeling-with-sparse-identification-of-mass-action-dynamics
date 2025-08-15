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
JLD2.@load "training_set_encoded.jld" ref_data_encoded ref_emit_encoded ref_data_min ref_data_max ref_params_train encoder
timelength = 60 * (ndays * 24) # minutes
dt = 60.0
startday = 2
times = LinRange(0, timelength, ntimestep)
tspan = (times[1], times[end])

n_latent_species = 6
seed=1234
Random.seed!(seed)
encoder = reshape(encoder,(:))
size_encoder = size(encoder)[1]
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


    dc_encoded = diff(ref_data_encoded;dims=2)./dt
    dc_encoded = cat(dc_encoded, dc_encoded[:,end:end,:];dims=2)
    dc_std = std(dc_encoded;dims=(2,3))
    dc_encoded = dc_encoded  ./ dc_std
    ref_emit_encoded = ref_emit_encoded ./dc_std

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
    umat = umat |> gpu
    rsys = ReactionSystem(basis; name=:simady)
    stoich = netstoichmat(rsys) |> gpu
    basis

    function oderatelaws(ξ, k, oneplusu, umat)
        ξabsmasked = abs.(ξ)
        ξmat = transpose(reshape(ξabsmasked, n_latent_rxn_constant, :))
        umat = reshape(umat, size(umat)[1], size(umat)[2], 1)
        oneplusu = reshape(oneplusu, 1, size(oneplusu)[1], size(oneplusu)[2])
        (ξmat * k) .* reshape(prod(oneplusu.^umat; dims=2),(size(umat)[1],:))
         
    end
    
    
    function dudt(ps, k, u, stoich, umat, weight)
        ξ_simady = ps
        u = reshape(u, (size(u)[1],:))
        k = reshape(k, (size(k)[1],:))
        oneplusu = [1; u]
        ratelaws = oderatelaws(ξ_simady, k, oneplusu, umat)
        (stoich * ratelaws)
    end

    
    batchsize = 30
    training_dataset = DataLoader((conc=cu(ref_data_encoded), emit=cu(ref_emit_encoded), dcdt=cu(dc_encoded), p = cu(k_params)), batchsize = 30)
    c0, e0, dcdt0, p0 = (first(training_dataset))
    c0 = c0 |> gpu
    e0 = e0 |> gpu
    dcdt0 = dcdt0 |> gpu
    p0 = p0 |> gpu

    i_ozone = 11
    # loss function
    function loss_func(x, ps, print_flag)
        c, e, dc, P = x
        sparse_coeff = ps
        pred_ = dudt(sparse_coeff, P, c, stoich, umat, [1.0, 0.0]) .+ reshape(e, n_latent_species, :)
        true_ = reshape((dc),(size(dc)[1],:))
        l_simady_norm = Flux.mse(pred_, true_)
    
        if print_flag == 1
            println("l_simady_norm = $(l_simady_norm)")
        end
        
        loss = ( 
         l_simady_norm
    )
        return loss
    end



    function thresholding(ξ, threshold)
        changed = false
        ξ =  ξ |> cpu
        ξ =  abs.(ξ)
        for i in 1:basis_size
            if ξ[i] < threshold
                ξ[i] = 0.0
            end
        end
        ξ =  ξ |> gpu
         return ξ
    end



    function train!(loss, ps, data)
   
        for (i, (C,E,dCdt,P)) in enumerate(data)
            #C,E,dCdt, P = cu(C), cu(E), cu(dCdt), cu(P)
            grad = Zygote.gradient(ps) do x
                loss_func([C,E,dCdt,P],x, 0)
            end 
            Flux.update!(opt_state, ps, grad[1])
    
            ps .= abs.(ps)
        end

    end

    seed=1234
    Random.seed!(seed)
    sparse_coeff = (ones(basis_size)).*1e-2
    ps = cu(sparse_coeff)

    learning_rate = 3e-3
    opt = Flux.ADAM(learning_rate)
    opt_state = Flux.setup(opt, ps) 

loss_history = Float32[]
loss_converged_steps = Int[]
global_epoch = 1
stage_converged = Dict{Int, Int}() 
open("model/simady/simady_output.txt", "w") do file
    write(file," ")
end
for stage = 1:5
    Optimisers.adjust!(opt_state, learning_rate * 10^(stage*(-1.0)+1))  # learning rate decay
        global ps
        for epoch in 1:300
            train!(loss_func, ps, training_dataset)
            loss_value = loss_func([c0, e0, dcdt0, p0],ps, 0)
            push!(loss_history, loss_value)
            print("stage $(stage): e = $(@sprintf("%d", epoch)), $(@sprintf("%.5E", loss_value));\n")
            open("model/simady/simady_output.txt", "a") do file
                write(file, "stage $(stage):e = $(@sprintf("%d", epoch)), $(@sprintf("%.4E", loss_value));\n")
            end
        end
        global ps = thresholding(ps, 1e-4)
        ps_cpu = cpu(vcat(encoder, cpu(ps)))
        num_zeros = count(==(0.0), ps_cpu[length(encoder)+1:end])
        println("zeros/all = $(num_zeros)/$(length(ps_cpu[length(encoder)+1:end]))")
        open("model/simady/simady_output.txt", "a") do file
                write(file, "zeros/all = $(num_zeros)/$(length(ps_cpu[length(encoder)+1:end]));\n")
        end
        JLD2.jldsave("model/simady/stage_$(@sprintf("%d", stage))_n_latent_species_$(n_latent_species).jld"; ps_cpu)

end

println(" ")