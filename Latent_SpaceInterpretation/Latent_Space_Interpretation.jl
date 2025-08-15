begin
    using Plots
    using Statistics, StatsBase
    using Printf
    using JLD2
    using CSV, DataFrames
    using LinearAlgebra
    using Random
    using Flux
end

cd(Base.source_path()*"/..")

ndays = 4

saveat = 60.0
dt = 60 # minutes
nruns = 4350
n_species = 5838
nmete = 5
ntimestep = 24*ndays +1

startspec = 1


timelength = 60 * (ndays * 24)
dt = 60.0
startday = 2
times = LinRange(0, timelength, ntimestep)
tspan = (times[1], times[end])


specname =  Array(CSV.read("specname.csv", DataFrame))[startspec:end,2]

discard_specname = vcat(string.(specname[1:60]),["H2SO4", "ETHP", "ANOL", "ETHOOH", "ALD2", "RCOOH", "C2O3", "ARO1", "ARO2", "ALK1", "OLE1", "API1", "API2", "LIM1", "LIM2", "PAR", "AONE", "MGLY", "ETH", "OLET", "OLEI", "TOL", "XYL", "CRES", "TO2", "CRO", "OPEN", "ONIT", "ROOH", "RO2", "ANO2", "NAP", "XO2", "XPAR", "ISOP", "ISOPRD", "ISOPP", "ISOPN", "ISOPO2", "API", "LIM", "CH3SO2H", "CH3SCH2OO", "CH3SO2OO", "CH3SO2CH2OO", "SULFHOX", "NA", "SA"])

indices_s = findall(x -> x in discard_specname, specname)
all_indices = collect(1:length(specname))
non_s_indices = setdiff(all_indices, indices_s)
specname = specname[non_s_indices]

n_latent_species = 6
seed=1234
Random.seed!(seed)
encoder = abs.(Flux.glorot_uniform(Random.seed!(seed), n_latent_species, n_species)) 
size_encoder = size(reshape(encoder,:))[1]
JLD2.@load "epoch_5000.jld"  ps_cpu

ps = (ps_cpu)
encoder, sparse_coeff = ps[1:size_encoder], ps[size_encoder+1:end]
encoder = reshape(encoder,(n_latent_species,:))

function latent_map(encoder,specname)
    value = similar(encoder)
    index = similar(encoder)
    comb = Matrix{Any}(undef, size(encoder,1), size(encoder,2)*2)
    for row in 1:size(encoder, 1)
        value[row, :] .= sort(encoder[row, :], rev=true)
        index[row, :] = sortperm(encoder[row, :], rev=true)
    end
        name = [specname[Int(element)] for element in index]
    for i in 1:size(encoder,2)
        comb[:, 2i-1] = name[:, i]
        comb[:, 2i] = value[:, i]
    end
    
    
    for i in 1:n_latent_species
        println("*latent species $(i)*:")
        sum_value = sum(value[i, :])
        for ii in 1:5
            value
            println("   $(name[i, ii]): $(@sprintf "%.5f" ((value[i, ii])/sum_value.*100))%")
            
        end
    end
    value, name, index, comb
end
encoder_value, encoder_name, encoder_index, comb = latent_map((abs.(encoder)), specname)
encoder_index_concerned = Int.(unique(encoder_index[:,1:5]))
ccc = cgrad([:white, cgrad(:roma, 10, categorical = true, scale = :exp)[8]])

a = (Array(encoder)'[encoder_index_concerned,:]'./sum(encoder;dims=2)[:])'.*100
a = Array(encoder)'[encoder_index_concerned,:]
gr()
heatplot = heatmap(a,
                   xticks=(1:20),yticks=(1:size(encoder_index_concerned)[1],specname[encoder_index_concerned]),
                   xtickfontsize = 18, ytickfontsize = 18, 
                   cticks = 0:0.05:1,
                   clims = (0,1),
                   xlabelfontsize = 25, ylabelfontsize = 25,
                   xlabel="Latent species", ylabel="MCM chemical species", 
                   right_margin = 15Plots.mm,
                   c = :viridis, 
                   margin = 10Plots.mm,
                   size=(1500,1200)
                   )
fontsize = 18
nnrow, nncol = size(a)[1], size(a)[2]

annotate!([(j, i, text((@sprintf "%.1E" a[i,j]), fontsize, :orange)) for i in 1:nnrow for j in 1:nncol])

savefig(heatplot, "Latent_species_CtrlExpr_latent_species_multi_label_$(n_latent_species).svg")

