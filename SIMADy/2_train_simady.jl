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
    pwd()
#################### Part 1: generate reference data ####################
ndays = 4

saveat = 60.0
dt = 60 # minutes
nruns = 4350
n_species = 5838
nmete = 5
ntimestep = 24*ndays +1

startspec = 1
JLD2.@load "training_set_encoded.jld" ref_data_encoded ref_emit_encoded ref_data_min ref_data_max ref_params_train encoder
#ref_data = abs.(ref_data)
#ref_emit = abs.(ref_emit)

timelength = 60 * (ndays * 24) # minutes
dt = 60.0
startday = 2
times = LinRange(0, timelength, ntimestep)
#times = LinRange(0 , timelength, Int((timelength) / dt) + 1)
tspan = (times[1], times[end])

n_latent_species = 6
seed=1234
Random.seed!(seed)
# initialize an encoder, create functions of encoding/decoding processes
#encoder = abs.(Flux.glorot_uniform(Random.seed!(seed), n_latent_species, n_species)) #|> Flux.gpu
encoder = reshape(encoder,(:))
size_encoder = size(encoder)[1]
#encoder = ones(n_latent_species, n_species) .* 1e-3 |> Flux.gpu
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

    #encoding concentration and emission

    dc_encoded = diff(ref_data_encoded;dims=2)./dt
    dc_encoded = cat(dc_encoded, dc_encoded[:,end:end,:];dims=2)
    dc_std = std(dc_encoded;dims=(2,3))
    dc_encoded = dc_encoded  ./ dc_std
    ref_emit_encoded = ref_emit_encoded ./dc_std

#n_latent_species = 5
n_latent_emit_species = n_latent_species

model_params = ref_params_train#ref_params

sza = model_params[4:4,:,:]
#sza1 = cat(sza[:,7:end,:],sza[:,1:6,:];dims=2)
#sza2 = cat(sza[:,13:end,:],sza[:,1:12,:];dims=2)
#T P SZA E
model_params = cat(model_params[2:2,:,:],model_params[5:5,:,:], model_params[4:4,:,:]; dims=1)


# Sympolics

## mete and emit
@parameters sza press tempk1 tempk2
@parameters emit[1:n_latent_emit_species]
#k = Num[sza; press; tempk/100.0; emit]
@parameters k[1:size(model_params)[1]]
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
k_params = cat(max.(cos.(model_params[3:3,:,:]), 0),
               model_params[2:2,:,:], 
               #(model_params[1:1,:,:].^2).*exp.(model_params[1:1,:,:].^-1),
               exp.(model_params[1:1,:,:].^-1),
               exp.((model_params[1:1,:,:].^-1).*(-1)),
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
    #basis = basis |> gpu
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
        ξ_simady = ps#[1:simady_basis_size]
        #ξ_emis = ps[simady_basis_size+1:simady_basis_size+emis_basis_size]
        #ξ, ps_nn = ps[1:basis_size], ps[basis_size+1:end]
        #nn = re(ps_nn)
        u = reshape(u, (size(u)[1],:))
        k = reshape(k, (size(k)[1],:))
        #k_emit = reshape(k_emit, (size(k_emit)[1],:))
        oneplusu = [1; u]
        ratelaws = oderatelaws(ξ_simady, k, oneplusu, umat)# for (col_k, col_u) in (eachcol(k), eachcol(oneplusu))]...;dims=2)
        #ξ_emis_mat = abs.(ξ_emis)#transpose(reshape(ξ_emis, n_latent_emit_species, n_latent_emit_species))
        #println(size(ξ_emis_mat))
        #println(size(k_emit))
        (stoich * ratelaws) #.+ (ξ_emis_mat .* k_emit)
    end

    
    batchsize = 30
    training_dataset = DataLoader((conc=cu(ref_data_encoded), emit=cu(ref_emit_encoded), dcdt=cu(dc_encoded), p = cu(k_params)), batchsize = 30)
    #training_dataset = Flux.Data.DataLoader((conc=(ref_data), emit=(ref_emit.*60.0), dcdt=(ref_dcdt), p = (k_params)), batchsize = batchsize)
    #println(size(training_dataset.data))
    c0, e0, dcdt0, p0 = (first(training_dataset))
    c0 = c0 |> gpu
    e0 = e0 |> gpu
    dcdt0 = dcdt0 |> gpu
    p0 = p0 |> gpu

    i_ozone = 11
    # loss function
    function loss_func(x, ps, print_flag)
    
        #read data from the two argument arrays
        c, e, dc, P = x
        sparse_coeff = ps
        #encoder, sparse_coeff = ps[1:size_encoder], ps[size_encoder+1:end]
        #encoder = reshape(encoder,(n_latent_species,:))
        
        #encoding concentration and emission
        #c = C#encoder_(encoder, C, batchsize)
        #e = E#encoder_(encoder, E, batchsize)
        #dc = diff(c;dims=2)./dt
        #dc = cat(dc, dc[:,end:end,:];dims=2) #.- e.*60
        #dc_std = std(dc;dims=(2,3))
        #dc = dCdt#dc  ./ dc_std
    
        
        #C_pred = decoder_(encoder, c, batchsize)
        #E_pred = decoder_(encoder, e, batchsize)
        #dcdt = encoder_(encoder, dCdt, batchsize)
        #dCdt_pred = decoder_(encoder, dcdt, batchsize)
    
    
        #l_ozone = Flux.mse(C[i_ozone,:,:], C_pred[i_ozone,:,:]) 
        #l_concerned = Flux.mse(C[concerned_species_index,:,:], C_pred[concerned_species_index,:,:]) 
        #l_all = Flux.mse(C[3:end,:,:], C_pred[3:end,:,:]) 
        #l_emit = Flux.mse(E_pred[3:end,:,:], E[3:end,:,:]) 
        #l_dcdt = Flux.mse(dCdt_pred[3:end,:,:], dCdt[3:end,:,:]) 
    
    
        pred_ = dudt(sparse_coeff, P, c, stoich, umat, [1.0, 0.0]) .+ reshape(e, n_latent_species, :)
        true_ = reshape((dc),(size(dc)[1],:))
        l_simady_norm = Flux.mse(pred_, true_)#
    
        
        #pred_mean = mean(reshape(pred_,(n_latent_species,:,batchsize));dims=(2))
        #true_mean = mean(reshape(true_,(n_latent_species,:,batchsize));dims=(2))
        #l_sindy_norm_mean = Flux.mse(pred_mean, true_mean)
        
        #pred_12 = reshape(pred_,(n_latent_species,:,batchsize))[:,1:1,:]
        #true_12 = reshape(true_,(n_latent_species,:,batchsize))[:,1:1,:]
        #l_sindy_norm_12 = Flux.mse(pred_12, true_12)
    
       # pred_125 = reshape(pred_,(n_latent_species,:,batchsize))[:,1:11,:]
       # true_125 = reshape(true_,(n_latent_species,:,batchsize))[:,1:11,:]
       # l_sindy_norm_125 = Flux.mse(pred_125, true_125)
    
       # latent_cov = covariance_matrix(reshape(c,(n_latent_species, :)))
       # l_encoder_penalty =  sum(abs.(latent_cov .- (Diagonal(latent_cov))))
    
        
        #l_ozone = l_ozone.* 1e6
        #l_concerned = l_concerned .* 1e6
        #l_all = l_all .* 1e5
        #l_emit = l_emit .* 1e7
        #l_encoder_penalty = l_encoder_penalty.* 1e0
        #l_simady_norm = l_simady_norm.* 1e2
        #l_sindy_norm_mean = l_sindy_norm_mean.* 1e0
        #l_sindy_norm_12 = l_sindy_norm_12 .* 1e0
        #l_sindy_norm_125 = l_sindy_norm_125 .* 1e0
    
        if print_flag == 1
        #println("loss_ozone = $(l_ozone)")
        #println("l_concerned = $(l_concerned )")
        #println("l_all = $(l_all )")
        #println("l_emit = $(l_emit )")
        #println("l_encoder_penalty = $(l_encoder_penalty)")
        println("l_simady_norm = $(l_simady_norm)")
        #println("l_sindy_norm_mean = $(l_sindy_norm_mean)")
        #println("l_sindy_norm_12 = $(l_sindy_norm_12)")
        #println("l_sindy_norm_125 = $(l_sindy_norm_125)")
        #println("loss = $(loss)")
        end
        
        loss = ( #l_ozone
        #+ l_concerned 
        #+ l_all
        #+ l_emit 
        #+ l_encoder_penalty
         l_simady_norm
        #+ l_sindy_norm_mean
        #+ l_sindy_norm_12
        #+ l_sindy_norm_125
        #+ l_simady
        #+ Flux.mse(C, C_pred) 
        #+ Flux.mse(E, E_pred) + penalty
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
            #ps[size_encoder+1:end] .= thresholding!(ps[size_encoder+1:end], 1e-6)
            #ps .= thresholding!(ps, 1e-4)
            #C,E,dCdt, P,encoder, sparse_coeff = nothing, nothing, nothing, nothing, nothing, nothing
            #GC.gc(true)
        end
        #ps
    end

    seed=1234
    Random.seed!(seed)
    sparse_coeff = (ones(basis_size)).*1e-2
    ps = cu(sparse_coeff)

    learning_rate = 3e-3
    opt = Flux.ADAM(learning_rate)
    opt_state = Flux.setup(opt, ps) 
    #train!(loss_func, ps, training_dataset)
loss_history = Float32[]
loss_converged_steps = Int[]
global_epoch = 1
stage_converged = Dict{Int, Int}()  # e.g., 1 => step 45
open("model/simady/simady_output.txt", "w") do file
    write(file," ")
end
for stage = 1:5
    Optimisers.adjust!(opt_state, learning_rate * 10^(stage*(-1.0)+1))  # learning rate decay
    #while true # Full SGD till nothing is thresholded
        global ps
        for epoch in 1:300 # Full SGD till params are not updated
            train!(loss_func, ps, training_dataset)
            loss_value = loss_func([c0, e0, dcdt0, p0],ps, 0)
            push!(loss_history, loss_value)
            print("stage $(stage): e = $(@sprintf("%d", epoch)), $(@sprintf("%.5E", loss_value));\n")
            open("model/simady/simady_output.txt", "a") do file
                write(file, "stage $(stage):e = $(@sprintf("%d", epoch)), $(@sprintf("%.4E", loss_value));\n")
            end
            #global global_epoch += 1
            #if norm(ps - ps_prev) < max(norm(ps_prev), 1e-8) * 1e-3 #* 10^(stage*(-1.0)+1)
            #    push!(loss_converged_steps, global_epoch)
            #    print("stage $(stage) loss fully converges;\n")
            #    open("model/simady/simady_output.txt", "a") do file
            #        write(file, "stage $(stage) loss fully converges;\n")
            #    end
            #    break
            #end
        end
        global ps = thresholding(ps, 1e-4)
        ps_cpu = cpu(vcat(encoder, cpu(ps)))
        num_zeros = count(==(0.0), ps_cpu[length(encoder)+1:end])
        println("zeros/all = $(num_zeros)/$(length(ps_cpu[length(encoder)+1:end]))")
        open("model/simady/simady_output.txt", "a") do file
                write(file, "zeros/all = $(num_zeros)/$(length(ps_cpu[length(encoder)+1:end]));\n")
        end
        JLD2.jldsave("model/simady/stage_$(@sprintf("%d", stage))_n_latent_species_$(n_latent_species).jld"; ps_cpu)
        # if changed == false
        #     stage_converged[stage] = global_epoch
        #     print("stage $(stage) parameters fully thresholded;\n")
        #     break
        # end
    #end
end

#plot(Array(ps))








    # for epoch in 1:50
        
    #     if epoch  == 1
    #         loss_value = loss_func([c0, e0, dcdt0, p0],ps, 0)
    #         print("e = $(@sprintf("%d", epoch)); $(@sprintf("%.5E", loss_value));\n")
            
            
    #         open("model/simady/simady_output.txt", "w") do file
    #             write(file, "epoch $epoch; $loss_value \n")
    #     end
    #     end
    #     train!(loss_func, ps, training_dataset)

    #     if epoch % 1 == 0
            
    #         loss_value = loss_func([c0, e0, dcdt0, p0],ps, 0)
    #         print("e = $(@sprintf("%d", epoch)); $(@sprintf("%.5E", loss_value) ); \n")
        
    #         open("model/simady/simady_output.txt", "a") do file
    #             write(file, "epoch $epoch; $loss_value \n")
    #         end

    #         if epoch % 200 == 0
    #             ps_cpu = cpu(vcat(encoder, cpu(ps)))
    #             JLD2.jldsave("model/simady/epoch_$(@sprintf("%d", epoch))_n_latent_species_$(n_latent_species)_loss_$(@sprintf("%.3E", loss_value)).jld"; ps_cpu)
    #         end
    #     end
    
    
    #     if epoch  == 2000
    #         #opt.eta = 3e-4
    #         Optimisers.adjust!(opt_state, 3e-4)
    #         println(opt_state)
    #     end
    #     if epoch  == 4000
    #         #opt.eta = 3e-5
    #         Optimisers.adjust!(opt_state, 3e-5)
    #         println(opt_state)
    #     end
    #     if epoch  == 5000
    #         ps_cpu = cpu(vcat(encoder, cpu(ps)))
    #             JLD2.jldsave("model/simady/epoch_$(@sprintf("%d", epoch)).jld"; ps_cpu)
    #     end
    # end


println(" ")