# ## Simple linear regression

# `MLJ` essentially serves as a unified path to many existing Julia packages each of which provides their own functionalities and models, with their own conventions.
#
# The simple linear regression demonstrates this.
# Several packages offer it (beyond just using the backslash operator): here we will use `MLJLinearModels` but we could also have used `GLM`, `ScikitLearn` etc.
#
# To load the model from a given package use `@load ModelName pkg=PackageName`

using MLJ
models()

filter(model) = model.is_pure_julia && model.is_supervised && model.prediction_type == :probabilistic
models(filter)
models("XGB")
measures("F1")

# mdls = models(matching(X, y))  # Commented out - X and y not defined yet

# Linear regression

LR = @load LinearRegressor pkg = MLJLinearModels

# Note: in order to be able to load this, you **must** have the relevant package in your environment, if you don't, you can always add it (``using Pkg; Pkg.add("MLJLinearModels")``).
#
# Let's load the _boston_ data set

import RDatasets: dataset
import DataFrames: describe, select, Not, rename!, DataFrame
data = dataset("MASS", "Boston")
println(first(data, 3))

# Let's get a feel for the data

@show describe(data)

# So there's no missing value and most variables are encoded as floating point numbers.
# In MLJ it's important to specify the interpretation of the features (should it be considered as a Continuous feature, as a Count, ...?), see also [this tutorial section](/getting-started/choosing-a-model/#data_and_its_interpretation) on scientific types.
#
# Here we will just interpret the integer features as continuous as we will just use a basic linear regression:

data = coerce(data, autotype(data, :discrete_to_continuous))

# Let's also extract the target variable (`MedV`):

y = data.MedV
X = select(data, Not(:MedV))

# Let's declare a simple multivariate linear regression model:

model = LR()

# First let's do a very simple univariate regression, in order to fit it on the data, we need to wrap it in a _machine_ which, in MLJ, is the composition of a model and data to apply the model on:

X_uni = select(X, :LStat) # only a single feature
mach_uni = machine(model, X_uni, y)
fit!(mach_uni)

# You can then retrieve the  fitted parameters using `fitted_params`:

fp = fitted_params(mach_uni)
@show fp.coefs
@show fp.intercept

# You can also visualise this

using Plots

plot(X.LStat, y, seriestype=:scatter, markershape=:circle, legend=false, size=(800, 600))

#  MLJ.predict(mach_uni, Xnew) to predict from a fitted model
Xnew = (LStat=collect(range(extrema(X.LStat)..., length=100)),)
plot!(Xnew.LStat, MLJ.predict(mach_uni, Xnew), linewidth=3, color=:orange)


# The  multivariate linear regression case is very similar

mach = machine(model, X, y)
fit!(mach)

fp = fitted_params(mach)
coefs = fp.coefs
intercept = fp.intercept
for (name, val) in coefs
    println("$(rpad(name, 8)):  $(round(val, sigdigits=3))")
end
println("Intercept: $(round(intercept, sigdigits=3))")

# You can use the `machine` in order to _predict_ values as well and, for instance, compute the root mean squared error:

ŷ = MLJ.predict(mach, X)
round(rsquared(ŷ, y), sigdigits=4)

# Let's see what the residuals look like

res = ŷ .- y
plot(res, line=:stem, linewidth=1, marker=:circle, legend=false, size=((800, 600)))
hline!([0], linewidth=2, color=:red)    # add a horizontal line at x=0
mean(y)

# Maybe that a histogram is more appropriate here

histogram(res, normalize=true, size=(800, 600), label="residual")


# ## Interaction and transformation
#
# Let's say we want to also consider an interaction term of `lstat` and `age` taken together.
# To do this, just create a new dataframe with an additional column corresponding to the interaction term:

X2 = hcat(X, X.LStat .* X.Age)

# So here we have a DataFrame with one extra column corresponding to the elementwise products between `:LStat` and `Age`.
# DataFrame gives this a default name (`:x1`) which we can change:

rename!(X2, :x1 => :interaction)

# Ok cool, now let's try the linear regression again

mach = machine(model, X2, y)
fit!(mach)
ŷ = MLJ.predict(mach, X2)
round(rsquared(ŷ, y), sigdigits=4)

# We get slightly better results but nothing spectacular.
#
# Let's get back to the lab where they consider regressing the target variable on `lstat` and `lstat^2`; again, it's essentially a case of defining the right DataFrame:

X3 = DataFrame(hcat(X.LStat, X.LStat .^ 2), [:LStat, :LStat2])
mach = machine(model, X3, y)
fit!(mach)
ŷ = MLJ.predict(mach, X3)
round(rsquared(ŷ, y), sigdigits=4)

# fitting y=mx+c to X3 is the same as fitting y=mx2+c to X3.LStat => Polynomial regression

# which again, we can visualise:

Xnew = (LStat=Xnew.LStat, LStat2=Xnew.LStat .^ 2)

plot(X.LStat, y, seriestype=:scatter, markershape=:circle, legend=false, size=(800, 600))
plot!(Xnew.LStat, MLJ.predict(mach, Xnew), linewidth=3, color=:orange)



# TODO HW : Find the best model by feature selection; best model means highest R²

# Feature Selection using Forward Selection
println("\n=== Feature Selection using Forward Selection ===")

# Get all available feature names
all_features = names(X)
println("Available features: ", all_features)

# Initialize variables for forward selection
selected_features = String[]
remaining_features = copy(all_features)
best_r2_overall = 0.0
best_features_overall = String[]

println("\nForward Selection Process:")
println("Step | Added Feature | Selected Features | R² Score")
println("-" ^ 60)

step = 0
while !isempty(remaining_features)
    global step, best_r2_overall, best_features_overall
    step += 1
    best_r2_step = -Inf
    best_feature_step = ""
    
    # Try adding each remaining feature
    for feature in remaining_features
        test_features = vcat(selected_features, [feature])
        X_test = select(X, Symbol.(test_features))
        
        # Fit model with current feature combination
        mach_test = machine(model, X_test, y)
        fit!(mach_test, verbosity=0)  # verbosity=0 to suppress output
        ŷ_test = MLJ.predict(mach_test, X_test)
        r2_test = rsquared(ŷ_test, y)
        
        # Check if this is the best feature to add in this step
        if r2_test > best_r2_step
            best_r2_step = r2_test
            best_feature_step = feature
        end
    end
    
    # Add the best feature if it improves the overall R²
    if best_r2_step > best_r2_overall
        push!(selected_features, best_feature_step)
        filter!(f -> f != best_feature_step, remaining_features)
        best_r2_overall = best_r2_step
        best_features_overall = copy(selected_features)
        
        println("$(lpad(step, 4)) | $(rpad(best_feature_step, 13)) | $(rpad(join(selected_features, ", "), 17)) | $(round(best_r2_step, digits=6))")
    else
        # No improvement, stop the forward selection
        println("No improvement found. Stopping forward selection.")
        break
    end
end


println("\n=== Best Model Found ===")
println("Best features: ", best_features_overall)
println("Best R² score: ", round(best_r2_overall, digits=6))
println("Number of features: ", length(best_features_overall))

# Fit the final best model
X_best = select(X, Symbol.(best_features_overall))
mach_best = machine(model, X_best, y)
fit!(mach_best)

# Display the coefficients of the best model
fp_best = fitted_params(mach_best)
println("\n=== Best Model Coefficients ===")
for (name, val) in fp_best.coefs
    println("$(rpad(string(name), 10)): $(round(val, sigdigits=4))")
end
println("$(rpad("Intercept", 10)): $(round(fp_best.intercept, sigdigits=4))")

# Compare with the full model
println("\n=== Comparison with Full Model ===")
mach_full = machine(model, X, y)
fit!(mach_full)
ŷ_full = MLJ.predict(mach_full, X)
r2_full = rsquared(ŷ_full, y)

println("Full model (all features) R²: $(round(r2_full, digits=6))")
println("Best model ($(length(best_features_overall)) features) R²: $(round(best_r2_overall, digits=6))")
println("Improvement: $(round(best_r2_overall - r2_full, digits=6))")
println("Features reduced: $(length(all_features) - length(best_features_overall)) out of $(length(all_features))")

# Visualize feature importance by showing R² contribution
println("\n=== Feature Importance (Individual R² when used alone) ===")
individual_r2 = Dict{String, Float64}()
for feature in all_features
    X_single = select(X, Symbol(feature))
    mach_single = machine(model, X_single, y)
    fit!(mach_single, verbosity=0)
    ŷ_single = MLJ.predict(mach_single, X_single)
    r2_single = rsquared(ŷ_single, y)
    individual_r2[feature] = r2_single
end

# Sort features by individual R²
sorted_features = sort(collect(individual_r2), by=x->x[2], rev=true)
for (feature, r2) in sorted_features
    selected_marker = feature in best_features_overall ? " ✓" : ""
    println("$(rpad(feature, 10)): $(round(r2, digits=6))$selected_marker")
end
