using Flux, MLDatasets, OneHotArrays, Random, Statistics, Printf, Plots, StatsPlots
using Optimisers


Random.seed!(42)


function load_fashionmnist(; batchsize::Int=128)
    # Load FashionMNIST dataset
    train_data = MLDatasets.FashionMNIST(split=:train)
    test_data = MLDatasets.FashionMNIST(split=:test)
    
    # Preprocess training data
    train_x = Float32.(train_data.features) ./ 255.0f0  # Normalize to [0,1]
    train_x = reshape(train_x, 28*28, :)  # Flatten to 784 features
    train_y = onehotbatch(train_data.targets, 0:9)
    
    # Preprocess test data
    test_x = Float32.(test_data.features) ./ 255.0f0
    test_x = reshape(test_x, 28*28, :)
    test_y = onehotbatch(test_data.targets, 0:9)
    
    # Create data loaders
    train_loader = Flux.DataLoader((train_x, train_y), batchsize=batchsize, shuffle=true)
    test_loader = Flux.DataLoader((test_x, test_y), batchsize=1000, shuffle=false)
    
    return train_loader, test_loader
end


function create_mlp(hidden_size::Int)
    return Chain(
        Dense(784 => hidden_size, relu),
        Dense(hidden_size => 10),
        softmax
    )
end


function accuracy(model, data_loader)
    correct = 0
    total = 0
    for (x, y) in data_loader
        ŷ = model(x)
        pred_labels = onecold(ŷ, 0:9)
        true_labels = onecold(y, 0:9)
        correct += sum(pred_labels .== true_labels)
        total += length(true_labels)
    end
    return correct / total
end

function train_model!(model, train_loader, test_loader, epochs::Int, lr::Float64; lr_schedule=nothing)
    opt = Flux.Adam(lr)
    opt_state = Flux.setup(opt, model)
    
    train_accuracies = Float64[]
    test_accuracies = Float64[]
    
    for epoch in 1:epochs
    
        if lr_schedule !== nothing
            new_lr = lr_schedule(epoch)
            opt = Flux.Adam(new_lr)
            opt_state = Flux.setup(opt, model)
        end
        
      
        for (x, y) in train_loader
            loss, grads = Flux.withgradient(model) do m
                ŷ = m(x)
                Flux.crossentropy(ŷ, y)
            end
            Flux.update!(opt_state, model, grads[1])
        end
        
        train_acc = accuracy(model, train_loader)
        test_acc = accuracy(model, test_loader)
        
        push!(train_accuracies, train_acc)
        push!(test_accuracies, test_acc)
        
        @printf("Epoch %d/%d - Train Acc: %.4f - Test Acc: %.4f", epoch, epochs, train_acc, test_acc)
        if lr_schedule !== nothing
            current_lr = lr_schedule(epoch)
            @printf(" - LR: %.6f", current_lr)
        end
        println()
    end
    
    return train_accuracies, test_accuracies
end

#===== TASK 1: VARYING HIDDEN LAYER SIZES =====#

println("=" ^ 60)
println("TASK 1: Varying Hidden Layer Sizes")
println("=" ^ 60)

# Load data
train_loader, test_loader = load_fashionmnist(batchsize=128)

# Test different hidden layer sizes
hidden_sizes = [10, 20, 40, 50, 100, 300]
final_test_accuracies = Float64[]

for hidden_size in hidden_sizes
    println("\nTraining MLP with hidden size: $hidden_size")
    model = create_mlp(hidden_size)
    
    train_accs, test_accs = train_model!(model, train_loader, test_loader, 10, 0.001)
    final_test_acc = test_accs[end]
    push!(final_test_accuracies, final_test_acc)
    
    println("Final test accuracy: $(round(final_test_acc * 100, digits=2))%")
end

# Plot results
p1 = plot(hidden_sizes, final_test_accuracies .* 100, 
         marker=:circle, linewidth=2, markersize=6,
         xlabel="Hidden Layer Size", ylabel="Test Accuracy (%)",
         title="Test Accuracy vs Hidden Layer Size",
         legend=false, grid=true)
display(p1)

#===== TASK 2: IMPACT OF RANDOM INITIALIZATION =====#

println("\n" * "=" ^ 60)
println("TASK 2: Impact of Random Initialization (Hidden Size = 30)")
println("=" ^ 60)

# Fixed hidden size of 30, run 10 times with different initializations
initialization_accuracies = Float64[]

for run in 1:10
    println("\nRun $run/10 with different initialization")
    Random.seed!(run * 42)  # Different seed for each run
    
    model = create_mlp(30)
    train_accs, test_accs = train_model!(model, train_loader, test_loader, 10, 0.001)
    final_test_acc = test_accs[end]
    push!(initialization_accuracies, final_test_acc)
    
    println("Final test accuracy: $(round(final_test_acc * 100, digits=2))%")
end

# Calculate statistics
mean_acc = mean(initialization_accuracies)
std_acc = std(initialization_accuracies)

println("\nInitialization Analysis Results:")
println("Mean accuracy: $(round(mean_acc * 100, digits=2))%")
println("Standard deviation: $(round(std_acc * 100, digits=2))%")

# Plot results
p2 = scatter(1:10, initialization_accuracies .* 100, 
            xlabel="Run Number", ylabel="Test Accuracy (%)",
            title="Test Accuracy Fluctuations (Hidden Size = 30)",
            markersize=6, alpha=0.7, grid=true)
hline!([mean_acc * 100], linestyle=:dash, linewidth=2, 
       label="Mean: $(round(mean_acc * 100, digits=2))%")
display(p2)

#===== TASK 3: BATCH SIZE 32, 25 EPOCHS, DECAYING LR =====#

println("\n" * "=" ^ 60)
println("TASK 3: Batch Size 32, 25 Epochs, Decaying Learning Rate")
println("=" ^ 60)

# Load data with batch size 32
train_loader_32, test_loader_32 = load_fashionmnist(batchsize=32)

# Define exponential decay learning rate schedule
function exponential_decay_schedule(epoch)
    initial_lr = 0.001
    decay_rate = 0.95
    return initial_lr * (decay_rate ^ (epoch - 1))
end

# Train model
Random.seed!(42)
model_task3 = create_mlp(30) 
train_accs_task3, test_accs_task3 = train_model!(
    model_task3, train_loader_32, test_loader_32, 25, 0.001, 
    lr_schedule=exponential_decay_schedule
)

println("\nTask 3 Results:")
println("Final test accuracy: $(round(test_accs_task3[end] * 100, digits=2))%")

# Plot training progress
p3 = plot(1:25, [train_accs_task3 .* 100, test_accs_task3 .* 100],
         label=["Train" "Test"], linewidth=2,
         xlabel="Epoch", ylabel="Accuracy (%)",
         title="Training Progress (Batch Size 32, Decaying LR)",
         grid=true)
display(p3)

#===== TASK 4: GRID SEARCH OPTIMIZATION =====#

println("\n" * "=" ^ 60)
println("TASK 4: Grid Search Optimization")
println("=" ^ 60)

# Define parameter grids
batch_sizes = [16, 32, 64]
learning_rates = [0.0005, 0.001, 0.002]
lr_schedules = [
    (epoch) -> 0.001,  # Constant
    (epoch) -> 0.001 * (0.95 ^ (epoch - 1)),  # Exponential decay
    (epoch) -> 0.001 * (0.5 ^ ((epoch - 1) ÷ 10))  # Step decay every 10 epochs
]
schedule_names = ["Constant", "Exponential", "Step"]

best_accuracy = 0.0
best_params = nothing
grid_results = []

for (i, batch_size) in enumerate(batch_sizes)
    for (j, base_lr) in enumerate(learning_rates)
        for (k, lr_schedule) in enumerate(lr_schedules)
            println("\nGrid Search: Batch=$batch_size, LR=$base_lr, Schedule=$(schedule_names[k])")
            
            # Load data with current batch size
            train_loader_grid, test_loader_grid = load_fashionmnist(batchsize=batch_size)
            
            # Train model
            Random.seed!(42)  # Same seed for fair comparison
            model_grid = create_mlp(30)
            
            # Adjust base learning rate for the schedule
            adjusted_schedule = (epoch) -> lr_schedule(epoch) * (base_lr / 0.001)
            
            _, test_accs_grid = train_model!(
                model_grid, train_loader_grid, test_loader_grid, 15, base_lr,  # Reduced epochs for grid search
                lr_schedule=adjusted_schedule
            )
            
            final_acc = test_accs_grid[end]
            push!(grid_results, (batch_size, base_lr, schedule_names[k], final_acc))
            
            println("Final accuracy: $(round(final_acc * 100, digits=2))%")
            
            if final_acc > best_accuracy
                global best_accuracy = final_acc
                global best_params = (batch_size, base_lr, k, adjusted_schedule)
                println("*** NEW BEST PARAMETERS ***")
            end
        end
    end
end

println("\nGrid Search Results:")
println("Best accuracy: $(round(best_accuracy * 100, digits=2))%")
println("Best parameters: Batch size=$(best_params[1]), Learning rate=$(best_params[2]), Schedule=$(schedule_names[best_params[3]])")

#===== TASK 5: FINAL TRAINING WITH BEST PARAMETERS =====#

println("\n" * "=" ^ 60)
println("TASK 5: Final Training with Best Parameters")
println("=" ^ 60)

# Train with best parameters for full 25 epochs
train_loader_best, test_loader_best = load_fashionmnist(batchsize=best_params[1])

Random.seed!(42)
model_final = create_mlp(30)
train_accs_final, test_accs_final = train_model!(
    model_final, train_loader_best, test_loader_best, 25, best_params[2],
    lr_schedule=best_params[4]
)

task3_final_acc = test_accs_task3[end]
task5_final_acc = test_accs_final[end]

println("\nFinal Comparison:")
println("Task 3 final accuracy: $(round(task3_final_acc * 100, digits=2))%")
println("Task 5 final accuracy: $(round(task5_final_acc * 100, digits=2))%")
println("Improvement: $(round((task5_final_acc - task3_final_acc) * 100, digits=2))%")

if task5_final_acc > task3_final_acc
    println("Yes,  improved the result from Task 3!")
else
    println("No improvement over Task 3 result.")
end

# Plot final comparison
p4 = plot(1:25, [test_accs_task3 .* 100, test_accs_final .* 100],
         label=["Task 3 (Original)" "Task 5 (Optimized)"], linewidth=2,
         xlabel="Epoch", ylabel="Test Accuracy (%)",
         title="Comparison: Original vs Optimized Parameters",
         grid=true)
display(p4)

#===== SUMMARY PLOTS =====#

println("\n" * "=" ^ 60)
println("GENERATING SUMMARY PLOTS")
println("=" ^ 60)

# Create a comprehensive summary plot
p_summary = plot(layout=(2,2), size=(800, 600))

# Plot 1: Hidden layer size analysis
plot!(p_summary[1], hidden_sizes, final_test_accuracies .* 100, 
      marker=:circle, linewidth=2, markersize=4,
      xlabel="Hidden Layer Size", ylabel="Test Accuracy (%)",
      title="Task 1: Hidden Layer Size Analysis")

# Plot 2: Random initialization
scatter!(p_summary[2], 1:10, initialization_accuracies .* 100, 
         markersize=4, alpha=0.7,
         xlabel="Run Number", ylabel="Test Accuracy (%)",
         title="Task 2: Random Initialization Impact")
hline!(p_summary[2], [mean_acc * 100], linestyle=:dash, linewidth=1)

# Plot 3: Training progress Task 3
plot!(p_summary[3], 1:25, test_accs_task3 .* 100,
      linewidth=2, label="Decaying LR",
      xlabel="Epoch", ylabel="Test Accuracy (%)",
      title="Task 3: Training with Decaying LR")

# Plot 4: Final comparison
plot!(p_summary[4], 1:25, [test_accs_task3 .* 100, test_accs_final .* 100],
      label=["Original" "Optimized"], linewidth=2,
      xlabel="Epoch", ylabel="Test Accuracy (%)",
      title="Task 5: Original vs Optimized")

display(p_summary)

println("All tasks have been successfully implemented and executed.")
println("Check the plots above for visual results of each analysis.")
