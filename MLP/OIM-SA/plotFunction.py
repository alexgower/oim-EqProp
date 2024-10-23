import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob # To use UNIX shell style wildcards


def plot_accuracy(dataframe, idx, color):

    single_train_error_tab = np.array(dataframe['Train_Acc'].values.tolist())
    single_test_error_tab = np.array(dataframe['Test_Acc'].values.tolist())

    plt.plot(single_train_error_tab, '-', color = color, alpha = 0.8, label = str(idx))
    plt.plot(single_test_error_tab, '--', color = color, alpha = 0.5, label = str(idx))

    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epochs')

    plt.title('Train and Test Accuracy (averaged)')
    plt.legend()

    # Save as error.png
    plt.savefig(f'{idx}/singleAcc.png')
    plt.close()  # Close the figure to free up memory


    return single_train_error_tab, single_test_error_tab






def plot_loss(dataframe, idx, color):

    single_train_loss_tab = np.array(dataframe['Train_Loss'].values.tolist())
    single_test_loss_tab = np.array(dataframe['Test_Loss'].values.tolist())

    plt.plot(single_train_loss_tab, '-', color = color, alpha = 0.8, label = str(idx))
    plt.plot(single_test_loss_tab, '--', color = color, alpha = 0.5, label = str(idx))

    plt.ylabel('Loss (%)')
    plt.xlabel('Epochs')

    plt.title('Train and Test loss')
    plt.legend()

    # Save as loss.png
    plt.savefig(f'{idx}/singleLoss.png')
    plt.close()  # Close the figure to free up memory


    return single_train_loss_tab, single_test_loss_tab






# TODo remove unused arguments
def plot_mean_error(store_ave_train_error, store_ave_test_error, store_single_train_error, store_single_test_error):
    '''
    Plot mean train & test error with +/- std
    '''

    # TODO check that we want this removed
    # store_ave_train_error, store_ave_test_error = np.array(store_ave_train_error), np.array(store_ave_test_error)
    # mean_ave_train, mean_ave_test = np.mean(store_ave_train_error, axis = 0), np.mean(store_ave_test_error, axis = 0)
    # std_ave_train, std_ave_test = np.std(store_ave_train_error, axis = 0), np.std(store_ave_test_error, axis = 0)

    store_single_train_error, store_single_test_error = np.array(store_single_train_error), np.array(store_single_test_error)
    mean_single_train, mean_single_test = np.mean(store_single_train_error, axis = 0), np.mean(store_single_test_error, axis = 0)
    std_single_train, std_single_test = np.std(store_single_train_error, axis = 0), np.std(store_single_test_error, axis = 0)

    epochs = np.arange(0, len(store_single_test_error[0])) # = 0:1:len(store_ave_test_error[0])
    max_epoch = 50
    plt.figure()


    # TODO check that we want this removed
    #plt.plot(epochs, mean_ave_train, label = 'mean_ave_train_accuracy')
    #plt.fill_between(epochs, mean_ave_train[:39] - std_ave_train[:39], mean_ave_train[:39] + std_ave_train[:39], facecolor = '#b9f3f3')
    #plt.plot(epochs, mean_ave_test, label = 'mean_ave_test_accuracy')
    #plt.fill_between(epochs, mean_ave_test - std_ave_test, mean_ave_test + std_ave_test, facecolor = '#fadcb3')


    # TODO see why no error bars are plotted
    plt.plot(epochs[:max_epoch], mean_single_train[:max_epoch], '--', label = 'mean_single_train_accuracy')
    plt.fill_between(epochs[:max_epoch], mean_single_train[:max_epoch] - std_single_train[:max_epoch], mean_single_train[:max_epoch] + std_single_train[:max_epoch], facecolor = '#b9f3f3')

    plt.plot(epochs[:max_epoch], mean_single_test[:max_epoch], '--', label = 'mean_single_test_accuracy')
    plt.fill_between(epochs[:max_epoch], mean_single_test[:max_epoch]- std_single_test[:max_epoch], mean_single_test[:max_epoch] + std_single_test[:max_epoch], facecolor = '#fadcb3')

    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epochs')
    plt.title('Mean train and Test Accuracy with std')
    plt.legend()

    # Save as mean_error.png
    plt.savefig(f'meanAcc.png')
    plt.close()  # Close the figure to free up memory


    return  mean_single_train[-1], std_single_train[-1], mean_single_test[-1], std_single_test[-1]
    







def plot_mean_loss(store_train_loss, store_test_loss):
    '''
    Plot mean train & test loss with +/- std
    '''

    store_train_loss, store_test_loss = np.array(store_train_loss), np.array(store_test_loss)
    mean_train, mean_test = np.mean(store_train_loss, axis = 0), np.mean(store_test_loss, axis = 0)
    std_train, std_test = np.std(store_train_loss, axis = 0), np.std(store_test_loss, axis = 0)
    
    
    epochs = np.arange(0, len(store_test_loss[0]))
    
    
    plt.figure()
    plt.plot(epochs, mean_train, label = 'mean_train_loss')
    plt.fill_between(epochs, mean_train - std_train, mean_train + std_train, facecolor = '#b9f3f3')

    plt.plot(epochs, mean_test, label = 'mean_test_loss')
    plt.fill_between(epochs, mean_test - std_test, mean_test + std_test, facecolor = '#fadcb3')

    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title('Mean train and Test loss with std')
    plt.legend()

    # Save as mean_loss.png
    plt.savefig(f'meanLoss.png')
    plt.close()  # Close the figure to free up memory










# IF USING THIS SCRIPT AS A STANDALONE
if __name__ == '__main__':

    path = os.getcwd()
    prefix = '/'

    files = glob.glob('*')

    store_ave_train_error, store_single_train_error, store_train_loss = [], [], []
    store_ave_test_error, store_single_test_error, store_test_loss = [], [], []


    colormap = plt.cm.RdPu
    colors = [colormap(i) for i in np.linspace(0, 1, len(files)+5)]


    # Only get data from correct folders (remove other files from files list)
    print(files)
    files_to_keep = []
    for idx, simu in enumerate(files):
        name, extension = os.path.splitext(simu)
        if extension not in ['.py', '.png']:
            files_to_keep.append(simu)
    files = files_to_keep
    print(files)
    files = sorted(files, key=lambda x: (int(x.split('-')[-1])))
    print(files)




    plt.figure()
    for idx, simu in enumerate(files):
        name, extension = os.path.splitext(simu)
        if not extension == '.py' and not extension == '.png':
            # Read results.csv file as dataframe
            DATAFRAME = pd.read_csv(path + prefix + simu + prefix + 'results.csv', sep = ',', index_col = 0)

            
            # PLOT ERROR
            single_train_error_tab, single_test_error_tab = plot_accuracy(DATAFRAME, name, color = colors[-(idx+1)])
            store_single_train_error.append(single_train_error_tab)
            store_single_test_error.append(single_test_error_tab)


            # PLOT LOSS
            train_loss_tab, test_loss_tab = plot_loss(DATAFRAME, name, color = colors[-(idx+1)])
            store_train_loss.append(train_loss_tab)
            store_test_loss.append(test_loss_tab)

        else:
            pass


        

    mean_ave_train, std_ave_train, mean_ave_test, std_ave_test = plot_mean_error(store_ave_train_error, store_ave_test_error, store_single_train_error, store_single_test_error)
    plot_mean_loss(store_train_loss, store_test_loss)


    print("Avg train acc = " +str(mean_ave_train) + " ± " + str(std_ave_train) + " std")
    print("Avg test acc = " +str(mean_ave_test) + " ± " + str(std_ave_test) + " std")

