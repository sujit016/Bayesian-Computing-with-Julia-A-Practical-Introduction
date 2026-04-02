using Plots, Statistics
using Distributions, Random, LaTeXStrings
using DataFrames, StatsBase
β₀ = 3   # population intercept
β₁ = 1  # population slope
ϕ = 1
x = range(1, 5, length = 50)
Random.seed!(123)
y = β₀ .+ β₁ .*x .+ rand(Normal(0, ϕ), length(x))
data = DataFrame(x = x, y = y)

scatter(x, y, color = "red", xlabel = L"x", ylabel = L"y", 
    label = "")

# Prior for β₀
μ₀ = 2
τ₀ = 0.4
prior_β₀(x) = pdf(Normal(μ₀, sqrt(τ₀)), x)

# Prior for ϕ
α = 2
γ = 2
prior_ϕ(x) = pdf(InverseGamma(α, 1/γ), x)

# Prior for β₁
μ₁ = 2
τ₁= 0.5
prior_β₁(x) = pdf(Normal(μ₁, sqrt(τ₁)), x)

function likelihood(data, params)
    x = data[:, 1]
    y = data[:, 2]
    n = nrow(data)
    β₀ = params[1]
    β₁ = params[2]
    ϕ = params[3]
    log_lik = -n*log(sqrt(2π)) - (n/2)*log(ϕ) -
              sum((y .- β₀ .- β₁ .* x).^2) / (2 * ϕ)
    return exp(log_lik)
end


iteration = 100000 
post_β₀ = Vector{Float64}(undef, iteration)
post_β₁ = Vector{Float64}(undef, iteration)
post_ϕ = Vector{Float64}(undef, iteration)

# specifications of grid
step = 0.01
grid_β₀ = -5:step:10
grid_β₁ = -5:step:10
# Initialization of the posterior samples
post_β₀[1] = rand(Normal(μ₀, sqrt(τ₀)))
post_β₁[1] = rand(Normal(μ₁, sqrt(τ₁)))
post_ϕ[1] = rand(InverseGamma(α, 1/γ))



for i in 2:iteration
    tmp_post_β₀ = zeros(length(grid_β₀))
    for j in eachindex(grid_β₀)
        params = [grid_β₀[j], post_β₁[i-1], post_ϕ[i-1]]
        tmp_post_β₀[j] = likelihood(data, params) * prior_β₀(grid_β₀[j])
    end
    prob = tmp_post_β₀ ./ sum(tmp_post_β₀)
    post_β₀[i] = sample(grid_β₀, Weights(prob))
    tmp_post_β₁ = zeros(length(grid_β₁))
    for j in eachindex(grid_β₁)
        params = [post_β₀[i], grid_β₁[j], post_ϕ[i-1]]
        tmp_post_β₁[j] = likelihood(data, params) * prior_β₁(grid_β₁[j])
    end
    prob = tmp_post_β₁ ./ sum(tmp_post_β₁)
    post_β₁[i] = sample(grid_β₁, Weights(prob))
    shape = α + nrow(data)/2
    rate = (1/2) * sum((y .- post_β₀[i] .- post_β₁[i] .* x).^2) + γ
    post_ϕ[i] = rand(InverseGamma(shape, rate))
end

p1 = plot(post_β₀, color = "red", lw = 2, xlabel = "index", ylabel = L"β_0", 
    title = "Trace Plot of β₀", label = "")
p2 = plot(post_β₁, color = "red", lw = 2, xlabel = "index", ylabel = L"β_1", 
    title = "Trace Plot of β₁", label = "")
p3 = plot(post_ϕ, color = "red", lw = 2, xlabel = "index", ylabel = L"ϕ", 
    title = "Trace Plot of ϕ", label = "")
plot(p1, p2, p3, layout = (2, 2), size = (800,600))
## Plot of conditional posterior density functions of β₀, β₁, ϕ
cut = 0.7
burn_in = ceil(Int, iteration * cut)
u = post_β₀[burn_in:end]   
index = 1:16:length(u)
post_β₀_thining = u[index]
acf(post_β₀_thining, xlabel = "lag", ylabel = "ACF" , label = "", 
    title = "acf of posterior β₀")
u = post_β₁[burn_in:end]
index = 1:16:length(u)
post_β₁_thining = u[index]
acf(post_β₁_thining, xlabel = "lag", ylabel = "ACF" , label = "", 
        title = "acf of posterior β₁")
u = post_ϕ[burn_in:end]
index = 1:16:length(u)
post_ϕ_thining = u[index]
acf(post_ϕ_thining, seriestype = :stem, xlabel = "lag", ylabel = "ACF" , label = "", 
        title = "acf of posterior ϕ")
p1 = histogram(post_β₀_thining, normalize = true, xlabel = L"β_0", ylabel = "Density", 
    title = "Posterior of β₀", label = "")
CI_β₀ = quantile(post_β₀_thining, [2.5, 97.5] ./100)
vline!([CI_β₀], color = "red", lw = 2 ,label = "95% CI")
p2 = histogram(post_β₁_thining, normalize = true, xlabel = L"β_1", ylabel = "Density", 
    title = "Posterior of β₁", label = "")
CI_β₁ = quantile(post_β₁_thining, [2.5, 97.5] ./100)
vline!([CI_β₁], color = "red", lw = 2 ,label = "95% CI")
p3 = histogram(post_ϕ_thining, normalize = true, xlabel = L"ϕ", ylabel = "Density", 
    title = "Posterior of ϕ", label = "")
CI_ϕ = quantile(post_ϕ_thining, [2.5, 97.5] ./100)
vline!([CI_ϕ], color = "red", lw = 2 ,label = "95% CI")
mean_post_β₀ = mean(post_β₀_thining)
mean_post_β₁ = mean(post_β₁_thining)
shape = α + nrow(data)/2
rate = (1/2)*sum((y .- mean_post_β₀ .- mean_post_β₁ .*x) .^2) + γ
plot!(x->pdf(InverseGamma(shape, rate), x), color = "magenta", lw = 2, label = "true curve")
plot(p1, p2, p3, layout = (2, 2), size = (800,600))
