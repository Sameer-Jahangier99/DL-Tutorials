# ## Getting started
using MLJ
import RDatasets: dataset
import DataFrames: DataFrame, select, Not, describe
using Random
using StatsPlots

data = dataset("datasets", "USArrests")
names(data) |> println

# Let's have a look at the mean and standard deviation of each feature:

describe(data, :mean, :std) |> show

# Let's extract the numerical component and coerce

X = select(data, Not(:State))
X = coerce(X, :UrbanPop => Continuous, :Assault => Continuous)

# ## PCA pipeline
#
# PCA is usually best done after standardization but we won't do it here:

PCA = @load PCA pkg = MultivariateStats

pca_mdl = Standardizer |> PCA(variance_ratio=1)
pca = machine(pca_mdl, X)
fit!(pca)
PCA
W = MLJ.transform(pca, X)

# W is the PCA'd data; here we've used default settings for PCA:

schema(W).names

# Let's inspect the fit:

r = report(pca)
cumsum(r.pca.principalvars ./ r.pca.tvar)

# In the second line we look at the explained variance with 1 then 2 PCA features and it seems that with 2 we almost completely recover all of the variance.

# ## More interesting data...

# Instead of just playing with toy data, let's load the orange juice data and extract only the columns corresponding to price data:

data = dataset("ISLR", "OJ")

feature_names = [
    :PriceCH, :PriceMM, :DiscCH, :DiscMM, :SalePriceMM, :SalePriceCH,
    :PriceDiff, :PctDiscMM, :PctDiscCH,
]

X = select(data, feature_names)
y = select(data, :Purchase)

train, test = partition(eachindex(y.Purchase), 0.7, shuffle=true, rng=1515)

using StatsBase
countmap(y.Purchase)
# ### PCA pipeline

Random.seed!(1515)

SPCA = Pipeline(
    Standardizer(),
    PCA(variance_ratio=1.0),
)

spca = machine(SPCA, X)
fit!(spca)
W = MLJ.transform(spca, X)
names(W)

# What kind of variance can we explain?

rpca = report(spca).pca
cs = cumsum(rpca.principalvars ./ rpca.tvar)


# Let's visualise this

using Plots
begin
    Plots.bar(1:length(cs), cs, legend=false, size=((800, 600)), ylim=(0, 1.1))
    xlabel!("Number of PCA features")
    ylabel!("Ratio of explained variance")
    plot!(1:length(cs), cs, color="red", marker="o", linewidth=3)
end
# So 4 PCA features are enough to recover most of the variance.

### HW TODO - Test the performance using LogisticClassifier and compare the performance on PCA features and the original set of features

# Load the LogisticClassifier
LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels

# Let's use 4 PCA features since they explain most of the variance
spca_4 = Pipeline(
    Standardizer(),
    PCA(maxoutdim=4),
)

# Fit the 4-component PCA pipeline
spca_4_machine = machine(spca_4, X)
fit!(spca_4_machine, rows=train)

# Transform both training and test data
W_train = MLJ.transform(spca_4_machine, selectrows(X, train))
W_test = MLJ.transform(spca_4_machine, selectrows(X, test))

# Get the target variable for train/test splits
y_train = selectrows(y, train)
y_test = selectrows(y, test)

println("=== Performance Comparison: Original Features vs PCA Features ===")

# 1. Train LogisticClassifier on ORIGINAL features
println("\n1. Training on Original Features ($(length(feature_names)) features):")

# Create pipeline with standardization for original features
original_pipeline = Pipeline(
    Standardizer(),
    LogisticClassifier()
)

original_model = machine(original_pipeline, X, y.Purchase)
fit!(original_model, rows=train)

# Predictions on original features
y_pred_original = MLJ.predict_mode(original_model, selectrows(X, test))

# Calculate accuracy for original features
original_accuracy = mean(y_pred_original .== y_test.Purchase)
println("Original features accuracy: $(round(original_accuracy, digits=4))")

# 2. Train LogisticClassifier on PCA features
println("\n2. Training on PCA Features (4 features):")

pca_model = machine(LogisticClassifier(), W_train, y_train.Purchase)
fit!(pca_model)

# Predictions on PCA features
y_pred_pca = MLJ.predict_mode(pca_model, W_test)

# Calculate accuracy for PCA features
pca_accuracy = mean(y_pred_pca .== y_test.Purchase)
println("PCA features accuracy: $(round(pca_accuracy, digits=4))")

# 3. Detailed comparison
println("\n=== Detailed Performance Analysis ===")
println("Number of original features: $(length(feature_names))")
println("Number of PCA features: 4")
# Ensure we don't access out of bounds
if length(cs) >= 4
    println("Variance explained by 4 PCA components: $(round(cs[4], digits=4))")
else
    println("Variance explained by $(length(cs)) PCA components: $(round(cs[end], digits=4))")
end
println("Original features accuracy: $(round(original_accuracy, digits=4))")
println("PCA features accuracy: $(round(pca_accuracy, digits=4))")
println("Accuracy difference: $(round(original_accuracy - pca_accuracy, digits=4))")
println("Dimensionality reduction: $(round((1 - 4/length(feature_names)) * 100, digits=1))%")

# 4. Let's also try different numbers of PCA components
println("\n=== Testing Different Numbers of PCA Components ===")

pca_accuracies = Float64[]
# Limit to the actual number of components available
max_components = min(length(feature_names), length(cs))
n_components = collect(1:max_components)

for n in n_components
    # Create PCA pipeline with n components
    spca_n = Pipeline(
        Standardizer(),
        PCA(maxoutdim=n),
    )
    
    # Fit and transform
    spca_n_machine = machine(spca_n, X)
    fit!(spca_n_machine, rows=train)
    
    W_train_n = MLJ.transform(spca_n_machine, selectrows(X, train))
    W_test_n = MLJ.transform(spca_n_machine, selectrows(X, test))
    
    # Train classifier
    pca_model_n = machine(LogisticClassifier(), W_train_n, y_train.Purchase)
    fit!(pca_model_n)
    
    # Predict and calculate accuracy
    y_pred_n = MLJ.predict_mode(pca_model_n, W_test_n)
    accuracy_n = mean(y_pred_n .== y_test.Purchase)
    push!(pca_accuracies, accuracy_n)
    
    # Safe access to cs array
    variance_explained = n <= length(cs) ? cs[n] : cs[end]
    println("$(n) PCA components: accuracy = $(round(accuracy_n, digits=4)), variance explained = $(round(variance_explained, digits=4))")
end

# 5. Visualize the results
using Plots
begin
    p1 = plot(n_components, pca_accuracies, 
             marker=:circle, linewidth=2, markersize=4,
             xlabel="Number of PCA Components", 
             ylabel="Test Accuracy",
             title="Classification Accuracy vs Number of PCA Components",
             legend=false, size=(800, 400))
    
    # Add horizontal line for original features accuracy
    hline!([original_accuracy], linestyle=:dash, linewidth=2, 
           label="Original Features ($(length(feature_names)) features)")
    
    display(p1)
end

# 6. Summary and recommendations
println("\n=== Summary and Recommendations ===")
best_pca_idx = argmax(pca_accuracies)
best_pca_components = n_components[best_pca_idx]
best_pca_accuracy = pca_accuracies[best_pca_idx]

println("Best PCA performance: $(best_pca_components) components with $(round(best_pca_accuracy, digits=4)) accuracy")
println("Original features performance: $(round(original_accuracy, digits=4)) accuracy")

if best_pca_accuracy >= original_accuracy
    println("✅ PCA features perform as well as or better than original features!")
    println("   Recommendation: Use $(best_pca_components) PCA components for $(round((1 - best_pca_components/length(feature_names)) * 100, digits=1))% dimensionality reduction")
else
    accuracy_loss = original_accuracy - best_pca_accuracy
    println("⚠️  PCA features have $(round(accuracy_loss, digits=4)) lower accuracy than original features")
    if accuracy_loss < 0.02  # Less than 2% accuracy loss
        println("   Recommendation: The accuracy loss is minimal. Consider using $(best_pca_components) PCA components for efficiency")
    else
        println("   Recommendation: Accuracy loss may be significant. Consider using original features or more PCA components")
    end
end

println("\nDimensionality reduction achieved: $(round((1 - best_pca_components/length(feature_names)) * 100, digits=1))%")
println("Variance explained by best PCA model: $(round(cs[best_pca_components], digits=4))")