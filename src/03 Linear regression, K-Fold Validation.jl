# ## Getting started

using MLJ
import RDatasets: dataset
import DataFrames: DataFrame, select
auto = dataset("ISLR", "Auto")
y, X = unpack(auto, ==(:MPG))
train, test = partition(eachindex(y), 0.5, shuffle=true, rng=444)

# Note the use of `rng=` to seed the shuffling of indices so that the results are reproducible.

# ### Polynomial regression

# This tutorial introduces polynomial regression in a very hands-on way. A more
# programmatic alternative is to use MLJ's `InteractionTransformer`. Run
# `doc("InteractionTransformer")` for details.

LR = @load LinearRegressor pkg = MLJLinearModels

# In this part we only build models with the `Horsepower` feature.

using Plots

begin
    plot(X.Horsepower, y, seriestype=:scatter, legend=false, size=(800, 600))
    xlabel!("Horsepower")
    ylabel!("MPG")
end

# Let's get a baseline:

lm = LR()
mlm = machine(lm, select(X, :Horsepower), y)
fit!(mlm, rows=train)
mse = rms(MLJ.predict(mlm, rows=test), y[test])^2

# Note that we square the measure to  match the results obtained in the ISL labs where the mean squared error (here we use the `rms` which is the square root of that).
xx = (Horsepower=range(50, 225, length=100) |> collect,)
yy = MLJ.predict(mlm, xx)

begin
    plot(X.Horsepower, y, seriestype=:scatter, label="Data", legend=false, size=(800, 600))
    plot!(xx.Horsepower, yy, label="Fit", legend=:topright, linewidth=3, color=:orange)
    xlabel!("Horsepower")
    ylabel!("MPG")
end

# We now want to build three polynomial models of degree 1, 2 and 3 respectively; we start by forming the corresponding feature matrix:

hp = X.Horsepower
Xhp = DataFrame(hp1=hp, hp2=hp .^ 2, hp3=hp .^ 3)

# Now we  can write a simple pipeline where the first step selects the features we want (and with it the degree of the polynomial) and the second is the linear regressor:

LinMod = Pipeline(
    FeatureSelector(features=[:hp1]),
    LR()
)

# Then we can  instantiate and fit 3 models where we specify the features each time:

LinMod.feature_selector.features = [:hp1] # poly of degree 1
lr1 = machine(LinMod, Xhp, y) # poly of degree 1 (line)
fit!(lr1, rows=train)

LinMod.feature_selector.features = [:hp1, :hp2] # poly of degree 2
lr2 = machine(LinMod, Xhp, y)
fit!(lr2, rows=train)

LinMod.feature_selector.features = [:hp1, :hp2, :hp3] # poly of degree 3
lr3 = machine(LinMod, Xhp, y)
fit!(lr3, rows=train)

# Let's check the performances on the test set

get_mse(lr) = rms(MLJ.predict(lr, rows=test), y[test])^2

@show get_mse(lr1)
@show get_mse(lr2)
@show get_mse(lr3)

# Let's visualise the models

hpn = xx.Horsepower
Xnew = DataFrame(hp1=hpn, hp2=hpn .^ 2, hp3=hpn .^ 3)

yy1 = MLJ.predict(lr1, Xnew)
yy2 = MLJ.predict(lr2, Xnew)
yy3 = MLJ.predict(lr3, Xnew)

begin
    plot(X.Horsepower, y, seriestype=:scatter, label=false, size=(800, 600))
    plot!(xx.Horsepower, yy1, label="Order 1", linewidth=3, color=:orange,)
    plot!(xx.Horsepower, yy2, label="Order 2", linewidth=3, color=:green,)
    plot!(xx.Horsepower, yy3, label="Order 3", linewidth=3, color=:red,)
    xlabel!("Horsepower")
    ylabel!("MPG")
end

# ## K-Folds Cross Validation

#
# Let's crossvalidate over the degree of the polynomial.
#
# **Note**: there's a  bit of gymnastics here because MLJ doesn't directly support a polynomial regression; see our tutorial on [tuning models](/getting-started/model-tuning/) for a gentler introduction to model tuning.
# The gist of the following code is to create a dataframe where each column is a power of the `Horsepower` feature from 1 to 10 and we build a series of regression models using incrementally more of those features (higher degree):

Xhp = DataFrame([hp .^ i for i in 1:10], :auto)

cases = [[Symbol("x$j") for j in 1:i] for i in 1:10]
r = range(LinMod, :(feature_selector.features), values=cases)

tm = TunedModel(model=LinMod, ranges=r, resampling=CV(nfolds=10), measure=rms)
#train + test => give you 10 splits of train+test
# Now we're left with fitting the tuned model

mtm = machine(tm, Xhp, y)
fit!(mtm)


rep = report(mtm)
res = rep.plotting
rep.best_model

# So the conclusion here is that the ?th order polynomial does quite well.
#
# In ISL they use a different seed so the results are a bit different but comparable.

Xnew = DataFrame([hpn .^ i for i in 1:10], :auto)
yy5 = MLJ.predict(mtm, Xnew)

begin
    plot(X.Horsepower, y, seriestype=:scatter, legend=false, size=(800, 600))
    plot!(xx.Horsepower, yy5, color=:orange, linewidth=4, legend=false)
    xlabel!("Horsepower")
    ylabel!("MPG")
end


### Effect of different features
using DataFrames
LinMod = Pipeline(
    FeatureSelector(features=[:Nmae]),
    LR()
)
names_cols = names(select(X, Not(:Name)))

cases = [[Symbol(names_cols[i]) for i in 1:j] for j in 1:lastindex(names_cols)]
r = range(LinMod, :(feature_selector.features), values=cases)

tm = TunedModel(model=LinMod, ranges=r, resampling=CV(nfolds=10), measure=rms)

# Now we're left with fitting the tuned model

mtm = machine(tm, X, y)
fit!(mtm)
rep = report(mtm)

res = rep.plotting
rep.best_model

best_models_mse_mean = mean(rep.best_history_entry.per_fold[1])^2
best_models_mse_std = std(rep.best_history_entry.per_fold[1])^2

# In this case, the best model is the one that uses all the features.

#HW TODO - find if MSE reduces further if we take and hyperparamaters tune upto 10 powers of each feature!

### Solution: Polynomial features for ALL numerical features with hyperparameter tuning

# First, let's identify the numerical features (excluding the target MPG and non-numeric Name)
numerical_features = names(select(X, Not(:Name)))
println("Numerical features to expand: ", numerical_features)

# Create polynomial features up to degree 10 for each numerical feature
X_poly = DataFrame()
feature_names = String[]

for feature in numerical_features
    feature_data = X[!, feature]
    for power in 1:10
        col_name = "$(feature)_p$(power)"
        X_poly[!, col_name] = feature_data .^ power
        push!(feature_names, col_name)
    end
end

println("Created $(ncol(X_poly)) polynomial features")
println("First few feature names: ", feature_names[1:min(10, length(feature_names))])

# Create the pipeline for polynomial feature selection
PolyLinMod = Pipeline(
    FeatureSelector(features=[Symbol(feature_names[1])]),  # Start with first feature
    LR()
)

# Create different feature combinations for hyperparameter tuning
# We'll try combinations of increasing complexity - using same approach as original code
max_features_to_try = min(30, length(feature_names))  # Limit to avoid overfitting

# Create cases similar to the original working code pattern
poly_cases = Vector{Vector{Symbol}}()

# Strategy 1: Sequential addition (exactly like the working original code)
for i in 1:max_features_to_try
    push!(poly_cases, [Symbol(feature_names[j]) for j in 1:i])
end

# Strategy 2: Add strategic degree-based combinations
# All degree-1 features (first power of each original feature)
degree_1_features = Vector{Symbol}()
for feat in numerical_features
    push!(degree_1_features, Symbol("$(feat)_p1"))
end
push!(poly_cases, degree_1_features)

# All degree-1 and degree-2 features
degree_1_2_features = Vector{Symbol}()
for feat in numerical_features
    for p in 1:2
        push!(degree_1_2_features, Symbol("$(feat)_p$(p)"))
    end
end
push!(poly_cases, degree_1_2_features)

println("Total feature combinations to try: ", length(poly_cases))

# Set up the hyperparameter range (exactly like the working original)
r_poly = range(PolyLinMod, :(feature_selector.features), values=poly_cases)

# Create the tuned model with cross-validation
tm_poly = TunedModel(
    model=PolyLinMod, 
    ranges=r_poly, 
    resampling=CV(nfolds=10), 
    measure=rms,
    acceleration=CPUThreads()  # Use parallel processing if available
)

# Fit the tuned model
println("Starting hyperparameter tuning for polynomial features...")
mtm_poly = machine(tm_poly, X_poly, y)
fit!(mtm_poly)

# Get results
rep_poly = report(mtm_poly)
best_poly_mse_mean = mean(rep_poly.best_history_entry.per_fold[1])^2
best_poly_mse_std = std(rep_poly.best_history_entry.per_fold[1])^2

println("\n=== RESULTS COMPARISON ===")
println("Original best model (all features, degree 1):")
println("  MSE Mean: $(best_models_mse_mean)")
println("  MSE Std:  $(best_models_mse_std)")
println("\nPolynomial features model:")
println("  MSE Mean: $(best_poly_mse_mean)")
println("  MSE Std:  $(best_poly_mse_std)")
println("  Best features: $(rep_poly.best_model.feature_selector.features)")
println("  Number of features in best model: $(length(rep_poly.best_model.feature_selector.features))")

# Check if polynomial features reduce MSE
mse_improvement = best_models_mse_mean - best_poly_mse_mean
println("\nMSE Improvement: $(mse_improvement)")
if mse_improvement > 0
    println("✅ YES! Polynomial features reduce MSE by $(round(mse_improvement, digits=4))")
    percent_improvement = (mse_improvement / best_models_mse_mean) * 100
    println("   That's a $(round(percent_improvement, digits=2))% improvement!")
else
    println("❌ No significant improvement with polynomial features")
end

# Plot comparison of MSE across different feature combinations
using Plots
mse_history = [mean(entry.per_fold[1])^2 for entry in rep_poly.history]
plot(1:length(mse_history), mse_history, 
     title="MSE vs Feature Combination", 
     xlabel="Feature Combination Index", 
     ylabel="MSE (Cross-Validation)",
     linewidth=2,
     size=(800, 400))
hline!([best_models_mse_mean], label="Original Best MSE", linestyle=:dash, color=:red, linewidth=2)
plot!(legend=:topright)

# Question - How can we use linear regression for classification?
