import Pkg

Pkg.activate(".")
Pkg.instantiate()

using Images: load, Gray, channelview
using Plots
using Dates

include("toy.jl")
gr()
Random.seed!(1274)


path = joinpath(@__DIR__, "bathymetry_maps", "map_Firth_of_Lorn_200m.tif")
path = joinpath(@__DIR__, "bathymetry_maps", "map_channel_narrow.jpg")

bathymetry_map = channelview(Gray.(load(path))) * 100 .- 1

#path = joinpath(@__DIR__, "bathymetry_maps", "map_channel_maze.jpg")

#--------- basics
# Load bathymetry, negative values are treated as land:
#bathymetry_map = channelview(Gray.(load("bathymetry_maps/map_Firth_of_Lorn_200m.tif"))) * 100 .- 1;
bathymetry_int = extrapolate(interpolate(bathymetry_map, BSpline(Linear())),-1.0);

#For toy bathy
receiver1 = Receiver((x=10, y=45), k=30.0, dist=30.0)
receiver2 = Receiver((x=100, y=50), k=30.0, dist=30.0)
receiver3 = Receiver((x=110, y=140), k=30.0, dist=30.0)

#For real bathy
receiver1 = Receiver((x=100, y=100), k=30.0, dist=30.0)
receiver2 = Receiver((x=300, y=100), k=30.0, dist=30.0)
receiver3 = Receiver((x=200, y=135), k=30.0, dist=30.0)

#--------- bridges
#Building bridges from receiver1 to receiver 2, which is a way to get all (if n_bridges, tmax and sigma are big enough to explore all the channels) the plausible 
#paths that the fish may have followed:
tmax = 100      #100 for toy bathimetry, 50 for real bathy

n_bridges = 1000
bridges = []
failed_traj=0
n=n_bridges/2
for _ in (1:(n_bridges/2)) 
    path = simulate_drift_bridge(
        tmax;
        rec1=receiver1,
        rec2=receiver2,
        σ_step=3.0,
        bathymetry_int=bathymetry_int,
        min_depth=0.0
    )
    
    if path !== nothing
        push!(bridges, path)
    else
        failed_traj=failed_traj+1
    end
end
print("Number of failed trajectories for horizontal bridges: $failed_traj, over $n")

failed_traj=0


bridges = filter(!isnothing, bridges)  # keep just the valid trajectories
bridges_x = [[p.x for p in bridge] for bridge in bridges]
bridges_y = [[p.y for p in bridge] for bridge in bridges]


println("Profondità a receiver1 (", receiver1.x, ",", receiver1.y, "): ", get_depth((x=receiver1.x, y=receiver1.y), bathymetry_int))
println("Profondità a receiver2 (", receiver2.x, ",", receiver2.y, "): ", get_depth((x=receiver2.x, y=receiver2.y), bathymetry_int))
println("Profondità a receiver3 (", receiver3.x, ",", receiver3.y, "): ", get_depth((x=receiver3.x, y=receiver3.y), bathymetry_int))
#To plot all the bridges
#=plt = heatmap(bathymetry_map[end:-1:1,:],
              xlim=(0, 200), ylim=(0, 200),
              color=:blues,
              legend=false,
              title="Brownian bridges betw receiver1 e receiver2")
for i in 1:length(bridges)
    plot!(plt, bridges_x[i], bridges_y[i], lw=2)
end
plot!(plt, make_circle(receiver1.x, receiver1.y, receiver1.dist), lw=2, color=:green, label="Receiver 1")
plot!(plt, make_circle(receiver2.x, receiver2.y, receiver2.dist), lw=2, color=:green, label="Receiver 2")
plot!(plt, make_circle(receiver2.x, receiver2.y, receiver2.dist), lw=2, color=:green, label="Receiver 3")
display(plt)=#

bridge_mean_depths = [mean(get_depth((x = p.x, y = p.y), bathymetry_int) for p in bridge) for bridge in bridges]
 
s_init  = bridges[argmin(bridge_mean_depths)]
s_depth = bridges[argmax(bridge_mean_depths)]


#s_init = bridges[end]
#s_depth = bridges[1]

xs_d = [p.x for p in s_depth]
ys_d = [p.y for p in s_depth]

xs = [p.x for p in s_init]
ys = [p.y for p in s_init]

#scatter!(plt, [receiver1.x, receiver2.x], [receiver1.y, receiver2.y], color=:red, label="Receivers")

plt2 = heatmap(bathymetry_map[end:-1:1,:],
              #xlim=(0, 200), ylim=(0, 200),
              color=:blues,
              legend=false,
              title="Brownian bridges tra receiver1 e receiver2")
plot!(plt2, xs, ys, lw=3, color=:red, label="chosen trajectory")
plot!(plt2, xs_d, ys_d, lw=3, color=:black, label="goal trajectory")
plot!(plt2, make_circle(receiver1.x, receiver1.y, receiver1.dist), lw=2, color=:green, label="activation 1")
plot!(plt2, make_circle(receiver2.x, receiver2.y, receiver2.dist), lw=2, color=:green, label="activation 2")
plot!(plt2, make_circle(receiver3.x, receiver3.y, receiver3.dist), lw=2, color=:green, label="activation 3") #for real bathy
display(plt2)



"""plt_all_bridges = heatmap(bathymetry_map[end:-1:1, :],
                           #xlim=(0, 200), ylim=(0, 200),
                           color=:blues,
                           legend=true,
                           title="Tutti i bridges simulati ($(length(bridges)) traiettorie)")
for i in 1:length(bridges)
    plot!(plt_all_bridges, bridges_x[i], bridges_y[i], lw=1, alpha=0.3, color=:red, label="")
end
plot!(plt_all_bridges, make_circle(receiver1.x, receiver1.y, receiver1.dist), lw=2, color=:green, label="Receiver 1")
plot!(plt_all_bridges, make_circle(receiver2.x, receiver2.y, receiver2.dist), lw=2, color=:green, label="Receiver 2")
plot!(plt_all_bridges, make_circle(receiver3.x, receiver3.y, receiver3.dist), lw=2, color=:green, label="Receiver 3")
display(plt_all_bridges)
"""











#--------- data
# Accustic signals:
receivers = [receiver1, receiver2]
Yaccustic = build_Yaccustic_from_trajectory(s_depth, receivers)

#Depth signal:
Ydepth = Tuple{Int, Float64, DepthGauge}[] 
depthgauge = DepthGauge()

#"Geolocating Fish Using Hidden Markov Models and Data Storage Tags" uses uniform noise in [-10,10], and a "depth model" that is a gaussian with sigma 15
for (t, point) in enumerate(s_depth)
    # Get the depth from the bathymetry
    d = get_depth((x=point.x, y=point.y), bathymetry_int)
    noisy_d = d + rand(Uniform(-10, 10))   #-1,1  for toy bahtimetry, -10,10 for real bathymetry
    push!(Ydepth, (t+1, noisy_d, depthgauge))
end
# print the depth

println("min=$(minimum(bathymetry_map)), max=$(maximum(bathymetry_map)), mean=$(sum(bathymetry_map)/length(bathymetry_map))")

#For gaussian noise instead
#=sigma_noise=0.5
for (t, point) in enumerate(bridges[2])
    d = get_depth((x=point.x, y=point.y), bathymetry_int)
    noisy_d = d + randn() * sigma_noise
    push!(Ydepth, (t+1, noisy_d, depthgauge))
end=#

#--------- PT 
@info "Optimizing posterior for a MAP starting point…"              

#Setting the problem characteristics:
mapping = TransformVariables.as(Array, 
                                TransformVariables.as((x = TransformVariables.asℝ, y = TransformVariables.asℝ)),
                                tmax)
v_init = TransformVariables.inverse(mapping, s_init)

fish_lp = FishLogPotential(Ydepth, Yaccustic, bathymetry_int, mapping, v_init, bridges)
fish_ref = FishReferencePotential(bathymetry_int, mapping, v_init, bridges, Yaccustic)


# Grid search ranges
n_chains = 64 #the min has to be >='n_local_mpi_processes'
n_rounds = 10
n_repeats = 1  # Number of repetitions

# Store results
all_trajectories = []
all_logposteriors = []
logpost_cold_values = Float64[]
std_devs = Float64[]
prof=[]

# Placeholder Ydepth for scatter comparison
Ydepth_values = [y[2] for y in Ydepth] 


#=Pkg.add("PairPlots")
Pkg.add("CairoMakie")
using PairPlots
using CairoMakie=#

pt_results = pigeons(
    target        = fish_lp,
    reference     = fish_ref,
    seed          = n_repeats,  # vary seed for reproducibility
    n_rounds      = n_rounds,
    n_chains      = n_chains,
    checkpoint    = true,
    multithreaded = true,
    explorer      = SliceSampler(),#=AutoMALA(
                step_size            = 6.0,           # passo iniziale MALA
                base_n_refresh       = 13,     #13        # passi base per esplorazione
                exponent_n_refresh   = 0.5,           # scala con √dim
                default_autodiff_backend = AutoEnzyme()        # backend autodiff
            ),=#
    record        = [traces, online, round_trip, Pigeons.timing_extrema, Pigeons.allocation_extrema, index_process]
)
pt = pt_results
pt_samples = Chains(pt)
myplot3 = plot(pt.reduced_recorders.index_process, title="chains=$n_chains, rounds=$n_rounds")
folder_path = "images/index_process"
mkpath(folder_path)
#myplot4 = pairplot(pt_samples) 
#CairoMakie.save("pair_plot5.svg", myplot4)
cold_last_v = pt_samples.value[end, 1:2*tmax] |> vec
cold_last_S = TransformVariables.transform(mapping, cold_last_v)

push!(all_trajectories, cold_last_S)

logposteriors = [fish_lp(pt_samples.value[i, 1:2*tmax]) for i in 1:size(pt_samples.value, 1)]
push!(all_logposteriors, logposteriors)

# Log posterior of last cold chain sample
logpost_cold = fish_lp(cold_last_v)
push!(logpost_cold_values, logpost_cold)

# Depth measurement
YdepthPIGEONS = Tuple{Int, Float64, DepthGauge}[] 
for (t, point) in enumerate(cold_last_S)
    d = get_depth((x=point.x, y=point.y), bathymetry_int)
    push!(YdepthPIGEONS, (t+1,d, depthgauge))
end

profonditaP = [y[2] for y in YdepthPIGEONS]
push!(prof, profonditaP)
std_dev = std(profonditaP .- Ydepth_values[1:length(profonditaP)])
push!(std_devs, std_dev)



std_dev = std(profonditaP .- Ydepth_values[1:length(profonditaP)])
            push!(std_devs, std_dev)

            # --- Plot traiettoria iniziale vs traiettoria finale di Pigeons ---
            xs_p = [p.x for p in cold_last_S]
            ys_p = [p.y for p in cold_last_S]

            plt_traj = heatmap(bathymetry_map[end:-1:1, :],
                                #xlim=(0, 200), ylim=(0, 200),
                                color=:blues,
                                legend=true,
                                title="Traiettoria iniziale vs Pigeons (chains=$n_chains, rounds=$n_rounds)")
            plot!(plt_traj, xs, ys, lw=3, color=:red, label="traiettoria iniziale (s_init)")
            plot!(plt_traj, xs_d, ys_d ,lw=3, color=:green, label="traiettoria depth")
            plot!(plt_traj, xs_p, ys_p, lw=3, color=:orange, label="traiettoria Pigeons (cold_last)")
            plot!(plt_traj, make_circle(receiver1.x, receiver1.y, receiver1.dist), lw=2, color=:green, label="")
            plot!(plt_traj, make_circle(receiver2.x, receiver2.y, receiver2.dist), lw=2, color=:green, label="")
            display(plt_traj)

            folder_traj = "images/trajectories"
            mkpath(folder_traj)
            #savefig(plt_traj, joinpath(folder_traj, timestamp * "_rep$(rep).svg"))

            # --- Plot profondità: dati osservati vs traiettoria Pigeons ---
            plt_depth = plot(1:length(Ydepth_values), Ydepth_values,
                              lw=2, color=:black, label="profondità osservata (dati rumorosi)",
                              xlabel="t", ylabel="profondità",
                              title="Profondità: dati vs Pigeons (chains=$n_chains, rounds=$n_rounds)")
            plot!(plt_depth, 1:length(profonditaP), profonditaP,
                  lw=2, color=:orange, label="profondità traiettoria Pigeons")
            display(plt_depth)

            folder_depth = "images/depth"
            mkpath(folder_depth)
            #savefig(plt_depth, joinpath(folder_depth, timestamp * "_rep$(rep).svg"))
