import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def get_max_epochs(data_list):
    """Get the maximum number of epochs from non-empty data."""
    return max(len(data) for data in data_list if len(data) > 0)

def plot_individual_accuracies(store_single_train_error, store_single_test_error, folders):
    """Plot all individual run accuracies on the same graph with distinct colors and fixed y-axis."""
    plt.figure(figsize=(12, 6))
    
    # Use a combination of color maps to get more distinct colors
    colors = []
    colors.extend(plt.cm.Set1(np.linspace(0, 1, 9)))
    colors.extend(plt.cm.Set2(np.linspace(0, 1, 8)))
    colors.extend(plt.cm.Dark2(np.linspace(0, 1, 8)))
    colors.extend(plt.cm.Paired(np.linspace(0, 1, 12)))
    
    # Plot each run up to its own maximum epoch
    for idx, (train_acc, test_acc, folder) in enumerate(zip(store_single_train_error, store_single_test_error, folders)):
        if len(train_acc) == 0 or len(test_acc) == 0:
            print(f"Skipping empty data for folder {folder}")
            continue
            
        epochs = np.arange(len(train_acc))
        color = colors[idx % len(colors)]
        
        plt.plot(epochs, train_acc, '-', 
                color=color, alpha=0.8, 
                label=f'{folder} (train)')
        plt.plot(epochs, test_acc, '--', 
                color=color, alpha=0.5, 
                label=f'{folder} (test)')
    
    plt.ylim(0, 100)  # Set y-axis limits to 0-100%
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epochs')
    plt.title('Train and Test Accuracy for Individual Runs')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('individual_accuracies.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_accuracy(dataframe, idx, color):
    if dataframe.empty:
        return np.array([]), np.array([])
        
    single_train_error_tab = np.array(dataframe['Train_Acc'].values.tolist())
    single_test_error_tab = np.array(dataframe['Test_Acc'].values.tolist())

    epochs = np.arange(len(single_train_error_tab))
    plt.figure()
    plt.plot(epochs, single_train_error_tab, '-', color=color, alpha=0.8, label=str(idx))
    plt.plot(epochs, single_test_error_tab, '--', color=color, alpha=0.5, label=str(idx))

    plt.ylim(0, 100)  # Set y-axis limits to 0-100%
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epochs')
    plt.title('Train and Test Accuracy (averaged)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.savefig(f'{idx}/singleAcc.png')
    plt.close()

    return single_train_error_tab, single_test_error_tab

def plot_loss(dataframe, idx, color):
    if dataframe.empty:
        return np.array([]), np.array([])
        
    single_train_loss_tab = np.array(dataframe['Train_Loss'].values.tolist())
    single_test_loss_tab = np.array(dataframe['Test_Loss'].values.tolist())

    epochs = np.arange(len(single_train_loss_tab))
    plt.figure()
    plt.plot(epochs, single_train_loss_tab, '-', color=color, alpha=0.8, label=str(idx))
    plt.plot(epochs, single_test_loss_tab, '--', color=color, alpha=0.5, label=str(idx))

    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title('Train and Test loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.savefig(f'{idx}/singleLoss.png')
    plt.close()

    return single_train_loss_tab, single_test_loss_tab

def plot_mean_error(store_ave_train_error, store_ave_test_error, store_single_train_error, store_single_test_error):
    # Filter out empty arrays
    store_single_train_error = [arr for arr in store_single_train_error if len(arr) > 0]
    store_single_test_error = [arr for arr in store_single_test_error if len(arr) > 0]
    
    if not store_single_train_error or not store_single_test_error:
        print("No valid data for mean error plot")
        return 0, 0, 0, 0

    # Find the minimum length across all arrays
    min_length = min(len(arr) for arr in store_single_train_error)
    
    # Truncate arrays to minimum length
    store_single_train_error = [arr[:min_length] for arr in store_single_train_error]
    store_single_test_error = [arr[:min_length] for arr in store_single_test_error]
    
    # Convert to numpy arrays for calculations
    store_single_train_error = np.array(store_single_train_error)
    store_single_test_error = np.array(store_single_test_error)
    
    mean_single_train = np.mean(store_single_train_error, axis=0)
    mean_single_test = np.mean(store_single_test_error, axis=0)
    std_single_train = np.std(store_single_train_error, axis=0)
    std_single_test = np.std(store_single_test_error, axis=0)

    epochs = np.arange(min_length)
    plt.figure()

    plt.plot(epochs, mean_single_train, '--', label='mean_single_train_accuracy')
    plt.fill_between(epochs, mean_single_train - std_single_train, 
                    mean_single_train + std_single_train, facecolor='#b9f3f3')

    plt.plot(epochs, mean_single_test, '--', label='mean_single_test_accuracy')
    plt.fill_between(epochs, mean_single_test - std_single_test, 
                    mean_single_test + std_single_test, facecolor='#fadcb3')

    plt.ylim(0, 100)  # Set y-axis limits to 0-100%
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epochs')
    plt.title('Mean Train and Test Accuracy with std')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.savefig('meanAcc.png')
    plt.close()

    return (mean_single_train[-1], std_single_train[-1], 
            mean_single_test[-1], std_single_test[-1])

def plot_mean_loss(store_train_loss, store_test_loss):
    # Filter out empty arrays
    store_train_loss = [arr for arr in store_train_loss if len(arr) > 0]
    store_test_loss = [arr for arr in store_test_loss if len(arr) > 0]
    
    if not store_train_loss or not store_test_loss:
        print("No valid data for mean loss plot")
        return
    
    # Find the minimum length across all arrays
    min_length = min(len(arr) for arr in store_train_loss)
    
    # Truncate arrays to minimum length
    store_train_loss = [arr[:min_length] for arr in store_train_loss]
    store_test_loss = [arr[:min_length] for arr in store_test_loss]
    
    store_train_loss = np.array(store_train_loss)
    store_test_loss = np.array(store_test_loss)
    
    mean_train = np.mean(store_train_loss, axis=0)
    mean_test = np.mean(store_test_loss, axis=0)
    std_train = np.std(store_train_loss, axis=0)
    std_test = np.std(store_test_loss, axis=0)
    
    epochs = np.arange(min_length)
    
    plt.figure()
    plt.plot(epochs, mean_train, label='mean_train_loss')
    plt.fill_between(epochs, mean_train - std_train, mean_train + std_train, 
                    facecolor='#b9f3f3')

    plt.plot(epochs, mean_test, label='mean_test_loss')
    plt.fill_between(epochs, mean_test - std_test, mean_test + std_test, 
                    facecolor='#fadcb3')

    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title('Mean Train and Test Loss with std')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.savefig('meanLoss.png')
    plt.close()

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    store_ave_train_error, store_single_train_error, store_train_loss = [], [], []
    store_ave_test_error, store_single_test_error, store_test_loss = [], [], []

    colormap = plt.cm.RdPu
    
    # Look for all folders in the current directory
    folders = [f for f in os.listdir(script_dir) 
              if os.path.isdir(f) and os.path.exists(os.path.join(f, 'results.csv'))]
    # Sort folders alphabetically
    folders.sort()
    
    colors = [colormap(i) for i in np.linspace(0, 1, len(folders)+5)]
    
    print("Found folders:", folders)
    
    for idx, folder in enumerate(folders):
        results_path = os.path.join(script_dir, folder, 'results.csv')
        
        if os.path.exists(results_path):
            DATAFRAME = pd.read_csv(results_path, sep=',', index_col=0)
            
            single_train_error_tab, single_test_error_tab = plot_accuracy(DATAFRAME, folder, color=colors[-(idx+1)])
            store_single_train_error.append(single_train_error_tab)
            store_single_test_error.append(single_test_error_tab)
            
            train_loss_tab, test_loss_tab = plot_loss(DATAFRAME, folder, color=colors[-(idx+1)])
            store_train_loss.append(train_loss_tab)
            store_test_loss.append(test_loss_tab)
        else:
            print(f"Warning: results.csv not found in folder {folder}")

    # Plot individual accuracies
    plot_individual_accuracies(store_single_train_error, store_single_test_error, folders)
    
    # Plot mean error and loss
    mean_ave_train, std_ave_train, mean_ave_test, std_ave_test = plot_mean_error(
        store_ave_train_error, store_ave_test_error, 
        store_single_train_error, store_single_test_error
    )
    plot_mean_loss(store_train_loss, store_test_loss)

    print("Avg train acc = " + str(mean_ave_train) + " ± " + str(std_ave_train) + " std")
    print("Avg test acc = " + str(mean_ave_test) + " ± " + str(std_ave_test) + " std")