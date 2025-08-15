begin
    using SciMLSensitivity
    using OrdinaryDiffEq
    using Random
    using Plots
    using DataDrivenDiffEq, DataDrivenSparse
    using Statistics, StatsBase
    using Printf
    using Flux
    using JLD2
    using ComponentArrays
    using DataInterpolations
    using Zygote
    using KissSmoothing
    end
    
    using CSV
    using DataFrames
    using LinearAlgebra
    

    using CUDA
CUDA.allowscalar(false)
CUDA.memory_status() 


ndays = 4

saveat = 60.0
dt = 60 # minutes
nruns = 4350
n_species = 5838
nmete = 5
ntimestep = 24*ndays +1

startspec = 1

#training set is too large to be uploaded to the repo so we attach the encoded dataset to this folder
JLD2.@load "training_set.jld" ref_data_train ref_emit_train ref_params_train specname

timelength = 60 * (ndays * 24) # minutes
dt = 60.0
startday = 2
times = LinRange(0, timelength, ntimestep)

tspan = (times[1], times[end])

concerned_species_index = [4, 5, 6, 10, 13, 14]
ref_data_max = maximum(ref_data_train[:,:,:];dims=(2,3))
ref_data_min = minimum(ref_data_train[:,:,:];dims=(2,3))
ref_data_normalized = (ref_data_train[:,:,:] .- ref_data_min ) ./(ref_data_max.-ref_data_min)
replace!(ref_data_normalized, NaN => 0.0)

i_ozone = 10
n_latent_species = 2
seed=1234
Random.seed!(seed)

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


batchsize = 30
training_dataset = Flux.Data.DataLoader((conc=(ref_data_normalized), emit = (ref_emit_train)), batchsize = 30)
C0, E0 = (first(training_dataset))
C0 = cu(C0)
E0 = cu(E0)

encoder = abs.(Flux.glorot_uniform(Random.seed!(1), n_latent_species, n_species)) |> Flux.gpu

#https://en.wikipedia.org/wiki/Cross-correlation
function covariance_matrix(Z)
    Z_centered = Z .- mean(Z, dims=2)
    cov_matrix = (Z_centered * Z_centered') / (size(Z, 2) - 1)
    return cov_matrix
end


# loss function
function loss_func(x, ps, print_flag)

    C, E = x
    encoder = ps
    c = encoder_(encoder, C, batchsize)
    e = encoder_(encoder, E, batchsize)
    
    C_pred = decoder_(encoder, c, batchsize)
    E_pred = decoder_(encoder, e, batchsize)

    l_concerned = Flux.mse(C[concerned_species_index,:,:], C_pred[concerned_species_index,:,:]) 
    l_all = Flux.mse(C, C_pred) 
    l_emit = Flux.mse(E_pred, E) 

    latent_cov = covariance_matrix(reshape(c,(n_latent_species, :)))
    l_encoder_penalty =  sum(abs.(latent_cov .- (Diagonal(latent_cov))))

    
    l_concerned = l_concerned .* 1e2
    l_all = l_all
    l_emit = l_emit
    l_encoder_penalty = l_encoder_penalty


    if print_flag == 1
    print("l_concerned = $(@sprintf("%.3E", l_concerned))")
    print(" l_all = $(@sprintf("%.3E", l_all));")
    print(" l_emit = $(@sprintf("%.3E", l_emit));")
    print(" l_encoder_penalty = $(@sprintf("%.3E", l_encoder_penalty));\n")

    end
    
    loss = ( 
     l_concerned 
    + l_all
    + l_emit 
    + l_encoder_penalty

)


    return loss
end


function loss_func_output(x, ps)

    C, E = x
    encoder = ps
    c = encoder_(encoder, C, batchsize)
    e = encoder_(encoder, E, batchsize)
    
    C_pred = decoder_(encoder, c, batchsize)
    E_pred = decoder_(encoder, e, batchsize)

    l_concerned = Flux.mse(C[concerned_species_index,:,:], C_pred[concerned_species_index,:,:]) 
    l_all = Flux.mse(C, C_pred) 
    l_emit = Flux.mse(E_pred, E) 


    latent_cov = covariance_matrix(reshape(c,(n_latent_species, :)))
    l_encoder_penalty =  sum(abs.(latent_cov .- (Diagonal(latent_cov))))

    
    #l_ozone = l_ozone.* 1e2
    l_concerned = l_concerned.* 1e2
    l_all = l_all #.* 1e5
    l_emit = l_emit #.* 1e6
    l_encoder_penalty = l_encoder_penalty #.* 1e0
    #println("loss = $(loss)")
    #print("l_ozone = $(@sprintf("%.3E", l_ozone));")
    #print(" loss_ozone_nox = $(@sprintf("%.3E", l_ozone_nox));")
    print(" l_concerned = $(@sprintf("%.3E", l_concerned));")
    print(" l_all = $(@sprintf("%.3E", l_all));")
    print(" l_emit = $(@sprintf("%.3E", l_emit));")
    print(" l_encoder_penalty = $(@sprintf("%.3E", l_encoder_penalty));\n")

    
    #l_ozone = ("l_ozone = $(@sprintf("%.3E", l_ozone));")
    l_concerned = (" l_concerned = $(@sprintf("%.3E", l_concerned));")
    #println("l_concerned = $(l_concerned )")
    l_all = (" l_all = $(@sprintf("%.3E", l_all));")
    l_emit = (" l_emit = $(@sprintf("%.3E", l_emit));")
    l_encoder_penalty = (" l_encoder_penalty = $(@sprintf("%.3E", l_encoder_penalty));")


    
    loss = ( #l_ozone
     l_concerned 
    * l_all
    * l_emit 
    * l_encoder_penalty

)


    return loss
end


function train!(loss, ps, data)
   
    for (i, (C,E)) in enumerate(data)
        C = cu(C)
        E = cu(E)
        grad = Zygote.gradient(ps) do x
            loss_func([(C),(E)],x, 0)
        end 
        
        Flux.update!(opt, ps, grad[1])
        ps .= abs.(ps)
    end
end

opt = Flux.ADAM(3e-4)



for epoch in 1:1000
    if epoch  == 1
        print("e = $(@sprintf("%d", epoch)); ")
        loss_value = loss_func_output([C0, E0], encoder)
        
        open("model/encoder/encoder_output.txt", "w") do file
            write(file, "epoch $epoch; $loss_value \n")
    end
    end
    
    if epoch % 1 == 0
        print("e = $(@sprintf("%d", epoch)); ")
        loss_value = loss_func_output([C0, E0], encoder)
        
        open("model/encoder/encoder_output.txt", "a") do file
            write(file, "epoch $epoch; $loss_value \n")
    end
        if epoch % 200 == 0
            loss_value = loss_func([C0, E0], encoder, 0)
            encoder_cpu = cpu(encoder)
            c = encoder_(cpu(encoder), ref_data_normalized, nruns)
            dc = diff(c;dims=2)./dt
            dc = cat(dc, dc[:,end:end,:];dims=2) #.- e.*60
            dc_std = std(dc;dims=(2,3))
            JLD2.jldsave("model/encoder/epoch_$(@sprintf("%d", epoch))_n_latent_species_$(n_latent_species)_loss_$(@sprintf("%.3E", loss_value)).jld"; encoder_cpu, dc_std)
        end
    end
    
    train!(loss_func, encoder, training_dataset)

    if epoch  == 400
        opt.eta = 3e-5
    end
    if epoch  == 800
        opt.eta = 3e-6
    end

if epoch  == 1000
        encoder_cpu = cpu(encoder)
            c = encoder_(cpu(encoder), ref_data_normalized, nruns)
            dc = diff(c;dims=2)./dt
            dc = cat(dc, dc[:,end:end,:];dims=2) #.- e.*60
            dc_std = std(dc;dims=(2,3))
            JLD2.jldsave("model/encoder/epoch_$(@sprintf("%d", epoch)).jld"; encoder_cpu, dc_std)
    end
    #GC.gc(true)
end