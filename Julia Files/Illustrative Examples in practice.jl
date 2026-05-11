# Example - I (Poisson rate parameter with gamma prior
# figure 2.1
using Plots, Statistics, StatsBase
using LaTeXStrings, Distributions, Random
using KernelDensity, SpecialFunctions
plt = plot(layout = (2, 2), size = (800, 600))
α = 4
β = 1/2
λ = range(0.1, 5, length = 50)
n_vals = [10, 25, 50, 100]
for (idx, n) in enumerate(n_vals) 
    mse_λ_cap = λ ./n
    mse_λB_cap = (β^2*n .*λ .+ (α*β .- λ) .^2) ./(n*β + 1)^2
    plot!(λ, mse_λ_cap, color = "red", lw = 2, xlabel = L"λ", ylabel = "MSE", 
        label = L"MSE_{λ}\hat{λ}", subplot = idx, title = "n = $n")
    plot!(λ, mse_λB_cap, color = "blue", lw = 2, label = L"MSE_{λB}\hat{λ}",
        subplot = idx)
end
display(plt)

# figure 2.2
# sampling distibution of sample mean
n = 5  # sample size
λ = 2  # true mean value
Random.seed!(123)
rep = 100
mean_vals = zeros(rep)
for i in 1:rep 
    mean_vals[i] = mean(rand(Poisson(λ), n))
end
d = kde(mean_vals)  # compute kernel density
plot(d.x, d.density, color = "magenta", lw = 2, ls = :dash, xlabel = L"λ", 
    ylabel = "Density", label = L"f_{\bar{X}}(x)")
α = 18
β = 1/3  # hyperparameters
prior_density(x) =  exp(-x/β)*x^(α-1)/(β^α * gamma(α))
plot!(prior_density, 0, 13, color = "red", lw = 3, ls = :dash, label = L"π(λ)")
y = sum(rand(Poisson(λ), n))  # sufficient statistics
function posterior_density(x)
    (n+1/β)^(y+α)*exp(-x*(n+1/β))*x^(y+α-1)/gamma(y+α)
end
plot!(posterior_density, color = "blue", lw = 3, ls = :dash, label = L"π(λ|y)")

##  Example - II (Normal prior for normal mean)
σ = sqrt(1)  # population sd
μ = 3  # prior mean value
τ = sqrt(0.5)  # prior sd
n_vals = [4, 10, 20, 30]
plt = plot(layout = (2, 2), size = (800, 600))
for (idx, n) in enumerate(n_vals) 
    plot!(x->(σ^2/(σ^2 + n*τ^2)^2*(n*τ^4 + (μ-x)^2*σ^2)), 0, 6, 
        color = "blue", lw = 3, xlabel = L"θ", ylabel = "MSE", 
        title = "n = $n", ylims = (0, 0.3), label = L"MSE_{θ}\hat{θ}", 
        subplot = idx)
    hline!([σ^2/n], color = "grey", lw = 2, label = L"MSE_{θ}\hat{θ_B}", 
        ls = :dash,subplot = idx)
    scatter!([μ],[0], markersize = 10, subplot = idx, label = "")
    annotate!(μ + 0.8, 0.01, "μ", subplot = idx)
end
display(plt)

plt = plot(layout = (2, 2), size = (800, 600))
n_vals = [4, 10, 20, 30]
σ = sqrt(1)
τ_vals = sqrt.([0.5, 1, 2])
for (idx, n) in enumerate(n_vals) 
    for i in 1:length(τ_vals)
        τ = τ_vals[i]
        if i == 1
            plot!(x->(σ^2/(σ^2 + n*τ^2)^2*(n*τ^4 + (μ-x)^2*σ^2)), 0, 6, 
            color = i + 1, lw = 3, xlabel = L"θ", ylabel = "MSE", 
            title = "n = $n", ylims = (0, 0.5), label = "τ = $τ", 
            subplot = idx, ls = :dash)
            hline!([σ^2/n], color = "grey", lw = 2, label = "", 
            ls = :dash,subplot = idx)
            scatter!([μ],[0], markersize = 10, subplot = idx, label = "")
            annotate!(μ + 0.8, 0.01, "μ", subplot = idx)
        else
            plot!(x->(σ^2/(σ^2 + n*τ^2)^2*(n*τ^4 + (μ-x)^2*σ^2)), 0, 6, 
            color = i + 1, lw = 3, xlabel = L"θ", ylabel = "MSE", 
            title = "n = $n", ylims = (0, 0.3), label = "τ = $τ", 
            subplot = idx, ls = :dash)
        end
    end
end
display(plt)

#  Example - III (Beta prior for Bernoulli())
plt = plot(layout = (2, 3), size = (800, 600))
n_vals = [4, 25, 50, 100, 200, 400]
rep = 10^4
p = range(0.001, 0.99, length = 25)
for (idx, n) in enumerate(n_vals) 
    α = sqrt(n/4)
    β = sqrt(n/4)
    mse_p_cap = zeros(length(p))
    mse_pB_cap = zeros(length(p))
    for i in 1:length(p)
        p_cap = zeros(rep)
        pB_cap = zeros(rep)
        for j in 1:rep 
            d = rand(Bernoulli(p[i]), n)
            p_cap[j] = sum(d)/n
            pB_cap[j] = (sum(d) + α)/(α + β + n )
        end
        mse_p_cap[i] = mean((p_cap .- p[i]) .^2)
        mse_pB_cap[i] = mean((pB_cap .- p[i]) .^2)
    end
    plot!(p, mse_p_cap, color = "red", lw = 2, xlabel = L"p", ylabel = "MSE", 
        ls = :dash, label = L"MSE_{p}\hat{p}", subplot = idx)
    plot!(p, mse_pB_cap, color = "blue", lw = 2, label = L"MSE_{p}\hat{p_B}", 
        subplot = idx)
end
display(plt)


# Example - IV (Generalization of Hierarchical Bayes)
using Distributions
x = rand(Gamma(2, 1/3), 50)
fit0 = fit(Gamma, x)
histogram(x, normalize = true, xlabel = L"x", ylabel = "Density", bins = 8 ,label = "")
plot!(x->pdf(Gamma(fit0.α, fit0.θ), x), color = "red", lw = 2, label = "")


## Example - V (Gamma prior for Exponential rate parameter)
Random.seed!(123)
n = 50
α = 8
β = 4
λ = rand(Gamma(α, β))
y = rand(Exponential(1/λ), n)
p_α = α + n
p_β = (1/β) + n*mean(y)
p_mean = p_α/p_β
plt = plot(layout = (2, 2), size = (800, 600))
M = [100, 200, 500, 1000]
for (idx, m) in enumerate(M) 
    z1 = rand(Gamma(p_α, 1/p_β), m)
    histogram!(z1, normalize = true,  title = "m = $m", xlabel = L"λ", 
        ylabel = "Density", label = "", subplot = idx)
    plot!(x->pdf(Gamma(p_α, 1/p_β), x), color = "red", lw = 2, label = "", 
        subplot = idx)
    scatter!([λ], [0], color = "red", markersize = 10, label = "", subplot = idx)
end
display(plt)
