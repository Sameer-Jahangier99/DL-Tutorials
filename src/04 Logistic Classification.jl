# ## Stock market data

# Let's load the usual packages and the data
using MLJ
import RDatasets: dataset #import does not bring the package and its fucntions in the namespace
import DataFrames: DataFrame, describe, select, Not
import StatsBase: countmap, cor, var

smarket = dataset("ISLR", "Smarket")
@show size(smarket)
@show names(smarket)

# Since we often  want  to only show a few significant digits for the metrics etc, let's introduce a very simple function  that does that:
r3(x) = round(x, sigdigits=3)
r3(pi)
ϵ = 0.1

# Let's get a description too
println(describe(smarket, :mean, :std, :eltype))

# The target variable is `:Direction`:

y = smarket.Direction
X = select(smarket, Not(:Direction))

# We can compute all the pairwise correlations; we use `Matrix` so that the dataframe entries are considered as one matrix of numbers with the same type (otherwise `cor` won't work):

using Plots
# using Plots: text

function plotter(cr::Matrix{Float64}, cols::Vector{Symbol})::Nothing
    (n, m) = size(cr)

    heatmap([i > j ? NaN : cr[i, j] for i in 1:m, j in 1:n], fc=cgrad([:red, :white, :dodgerblue4]), clim=(-1.0, 1.0), xticks=(1:m, cols), xrot=90, yticks=(1:m, cols), yflip=true, dpi=300, size=(800, 700), title="Pearson Correlation Coefficients")

    annotate!([(j, i, text(round(cr[i, j], digits=3), 10, "Computer Modern", :black)) for i in 1:n for j in 1:m])

    savefig("./pearson_correlations.png")
    return nothing
end

cm = X |> Matrix |> cor
plotter(round.(cm, sigdigits=1), Symbol.(names(X)))

# Let's see what the `:Volume` feature looks like:

using Plots

begin
    plot(X.Volume, size=(800, 600), linewidth=2, legend=false)
    xlabel!("Tick number")
    ylabel!("Volume")
end


# ### Logistic Regression

# We will now try to train models; the target `:Direction` has two classes: `Up` and `Down`; it needs to be interpreted as a categorical object, and we will mark it as a _ordered factor_ to specify that 'Up' is positive and 'Down' negative (for the confusion matrix later):

y = coerce(y, OrderedFactor)
levels(y)

# Note that in this case the default order comes from the lexicographic order which happens  to map  to  our intuition since `D`  comes before `U`.

cm = countmap(y)
categories, vals = collect(keys(cm)), collect(values(cm))
Plots.bar(categories, vals, title="Bar Chart Example", legend=false)
ylabel!("Number of occurrences")

# Seems pretty balanced.

# Let's now try fitting a simple logistic classifier (aka logistic regression) not using `:Year` and `:Today`:

LogisticClassifier = @load LogisticClassifier pkg = MLJLinearModels
X2 = select(X, Not([:Year, :Today]))
classif = machine(LogisticClassifier(), X2, y)

# Let's fit it to the data and try to reproduce the output:

fit!(classif)
ŷ = MLJ.predict(classif, X2)
ŷ[1:3]

# Note that here the `ŷ` are _scores_.
# We can recover the average cross-entropy loss:
cross_entropy(ŷ, y) |> r3

# in order to recover the class, we could use the mode and compare the misclassification rate:

ŷ = predict_mode(classif, X2)
misclassification_rate(ŷ, y) |> r3

# Well that's not fantastic...
#
# Let's visualise how we're doing building a confusion matrix,
# first is predicted, second is truth:

@show cm = confusion_matrix(ŷ, y)

# We can then compute the accuracy or precision, etc. easily for instance:

@show false_positive(cm)
@show accuracy(ŷ, y) |> r3
@show accuracy(cm) |> r3  # same thing
@show ppv(cm) |> r3   # a.k.a. precision
@show recall(cm) |> r3
@show f1score(ŷ, y) |> r3
@show fpr(ŷ, y) |> r3

# Let's now train on the data before 2005 and use it to predict on the rest.
# Let's find the row indices for which the condition holds

train = 1:findlast(X.Year .< 2005)
test = last(train)+1:length(y)

# We can now just re-fit the machine that we've already defined just on those rows and predict on the test:

fit!(classif, rows=train)
ŷ = predict_mode(classif, rows=test)
accuracy(ŷ, y[test]) |> r3

# Well, that's not very good...
# Let's retrain a machine using only `:Lag1` and `:Lag2`:

X3 = select(X2, [:Lag1, :Lag2])
classif = machine(LogisticClassifier(), X3, y)
fit!(classif, rows=train)
ŷ = predict_mode(classif, rows=test)
accuracy(ŷ, y[test]) |> r3

# Interesting... it has higher accuracy than the model with more features! This could be investigated further by increasing the regularisation parameter but we'll leave that aside for now.
#
# We can use a trained machine to predict on new data:

Xnew = (Lag1=[1.2, 1.5], Lag2=[1.1, -0.8])
ŷ = MLJ.predict(classif, Xnew)
ŷ |> println

# **Note**: when specifying data, we used a simple `NamedTuple`; we could also have defined a dataframe or any other compatible tabular container.
# Note also that we retrieved the raw predictions here i.e.: a score for each class; we could have used `predict_mode` or indeed

mode.(ŷ)


#HW TODO - Evaluate your LogisticClassifier using 10-folds

# Solution: 10-Fold Cross-Validation Evaluation
println("\n=== 10-Fold Cross-Validation Evaluation ===")

# Create a fresh machine with the best feature set (Lag1 and Lag2)
X3 = select(X2, [:Lag1, :Lag2])
classif_cv = machine(LogisticClassifier(), X3, y)

# Set up 10-fold cross-validation
cv = CV(nfolds=10, shuffle=true, rng=1234)  # Set rng for reproducibility

# Evaluate using cross-validation with compatible metrics
# Use only metrics that work reliably with MLJ
cv_results = evaluate!(classif_cv, 
                      resampling=cv,
                      measures=[accuracy, log_loss, f1score],
                      verbosity=1)

# Display results
println("\n10-Fold Cross-Validation Results:")
println("================================")
println("Accuracy: $(r3(cv_results.measurement[1])) ± $(r3(std(cv_results.per_fold[1])))")
println("Log Loss: $(r3(cv_results.measurement[2])) ± $(r3(std(cv_results.per_fold[2])))")
println("F1 Score: $(r3(cv_results.measurement[3])) ± $(r3(std(cv_results.per_fold[3])))")

# Calculate additional metrics manually from confusion matrix per fold
println("\nDetailed metrics per fold:")
cv_detailed = evaluate!(classif_cv,
                       resampling=cv, 
                       measures=[confusion_matrix],
                       verbosity=0)

# Calculate precision and recall from confusion matrices
precisions = []
recalls = []
for cm in cv_detailed.per_fold[1]
    # Extract values from confusion matrix
    tp = cm[2,2]  # True positives (Up predicted as Up)
    fp = cm[2,1]  # False positives (Down predicted as Up)  
    fn = cm[1,2]  # False negatives (Up predicted as Down)
    precision_val = tp / (tp + fp)
    recall_val = tp / (tp + fn)
    
    push!(precisions, precision_val)
    push!(recalls, recall_val)
end

println("Precision: $(r3(mean(precisions))) ± $(r3(std(precisions)))")
println("Recall: $(r3(mean(recalls))) ± $(r3(std(recalls)))")

# Also evaluate the model with all features (except Year and Today) for comparison
println("\n=== Comparison: All Features vs Selected Features ===")

# Model with all features (X2)
classif_all = machine(LogisticClassifier(), X2, y)
cv_results_all = evaluate!(classif_all, 
                          resampling=cv,
                          measures=[accuracy, log_loss, f1score],
                          verbosity=0)

println("\nAll Features (Lag1, Lag2, Lag3, Lag4, Lag5, Volume):")
println("  Accuracy: $(r3(cv_results_all.measurement[1])) ± $(r3(std(cv_results_all.per_fold[1])))")
println("  Log Loss: $(r3(cv_results_all.measurement[2])) ± $(r3(std(cv_results_all.per_fold[2])))")
println("  F1 Score: $(r3(cv_results_all.measurement[3])) ± $(r3(std(cv_results_all.per_fold[3])))")

println("\nSelected Features (Lag1, Lag2 only):")
println("  Accuracy: $(r3(cv_results.measurement[1])) ± $(r3(std(cv_results.per_fold[1])))")
println("  Log Loss: $(r3(cv_results.measurement[2])) ± $(r3(std(cv_results.per_fold[2])))")
println("  F1 Score: $(r3(cv_results.measurement[3])) ± $(r3(std(cv_results.per_fold[3])))")

# Statistical significance test (rough approximation)
accuracy_diff = cv_results.measurement[1] - cv_results_all.measurement[1]
println("\nAccuracy difference (Selected - All): $(r3(accuracy_diff))")
if abs(accuracy_diff) > 0.01
    println("Notable difference in performance between feature sets")
else
    println("Similar performance between feature sets")
end