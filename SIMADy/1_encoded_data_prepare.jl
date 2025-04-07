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

    using CSV
    using DataFrames
    using LinearAlgebra
end
        
    #using CUDA
    #CUDA.allowscalar(false)


ndays = 4

saveat = 60.0
dt = 60 # minutes
nruns = 4350
n_species = 5838
nmete = 5
ntimestep = 24*ndays +1

startspec = 1

JLD2.@load "training_set.jld" ref_data_train ref_emit_train ref_params_train specname
ref_data = abs.(ref_data_train)
ref_emit = abs.(ref_emit_train)

timelength = 60 * (ndays * 24) # minutes
dt = 60.0
startday = 2
times = LinRange(0, timelength, ntimestep)
#times = LinRange(0 , timelength, Int((timelength) / dt) + 1)
tspan = (times[1], times[end])



specname =  Array(CSV.read("specname.csv", DataFrame))[startspec:end,2]
specname = specname[61:end]

ref_data_max = maximum(ref_data_train[:,:,:];dims=(2,3))
ref_data_min = minimum(ref_data_train[:,:,:];dims=(2,3))
ref_data_normalized = (ref_data_train[:,:,:] .- ref_data_min ) ./(ref_data_max.-ref_data_min)
replace!(ref_data_normalized, NaN => 0.0)

encoder_file = "epoch_1000.jld"

    #for (i, i_spec) in  enumerate(2:10)

        n_latent_species = 2
        
        JLD2.@load "model/encoder/"*encoder_file  encoder_cpu dc_std
        
        encoder = (encoder_cpu)
        
        
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
        encoder
        ref_data_encoded = encoder_(encoder, ref_data_normalized, nruns)
            ref_emit_encoded = encoder_(encoder, ref_emit_train, nruns)
            JLD2.jldsave("training_set_encoded.jld"; ref_data_encoded, ref_emit_encoded, ref_data_min, ref_data_max, ref_params_train, encoder)
