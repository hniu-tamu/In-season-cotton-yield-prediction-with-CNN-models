import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os
import numpy as np
import pickle


def plot_performance(test_labels, y_pred, title = 'CNN', save_path = None):
    """
    Plot model performance with scatter plot of observed vs. predicted values.
    
    Parameters:
        test_labels (array-like): Observed values.
        y_pred (array-like): Predicted values.
    """
    r2 = r2_score(test_labels, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(test_labels, y_pred, color='black', s=13)
    ax.plot([0, 100], [0, 100], '--', color='k')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, )
    
    ax.set_title(f'{title} model performance ($R^2$: {r2:.2f})', fontsize=18)
    ax.set_xlabel('Observed yield (lb)', fontsize=18)
    ax.set_ylabel('Predicted yield (lb)', fontsize=18)
    ax.tick_params(labelsize=18)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved at: {save_path}")
    
    plt.show()

def plot_performance_2(test_labels, y_pred, title = 'CNN', save_path = None):
    """
    Plot model performance with scatter plot of observed vs. predicted values.
    
    Parameters:
        test_labels (array-like): Observed values.
        y_pred (array-like): Predicted values.
    """
    r2 = r2_score(test_labels, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(test_labels, y_pred, color='black', s=13)
    ax.plot([0, 1.4], [0, 1.4], '--', color='k')
    ax.set_xlim(0, 1.4)
    ax.set_ylim(0, 1.4)
    
    ax.set_title(f'{title} model performance ($R^2$: {r2:.2f})', fontsize=18)
    ax.set_xlabel('Observed yield (lb)', fontsize=18)
    ax.set_ylabel('Predicted yield (lb)', fontsize=18)
    ax.tick_params(labelsize=18)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved at: {save_path}")
    
    plt.show()


def load_cotton_data(dataset_folder, data_filename, yield_filename):
    """
    Load data from files located in a specified dataset folder.
    
    Parameters:
        dataset_folder (str): Name of the dataset folder.
        data_filename (str): Name of the file containing the image data.
        yield_filename (str): Name of the file containing yield data.
        
    Returns:
        tuple: A tuple containing the loaded image data and yield data.
    """
    # Construct file paths
    data_file_path = os.path.join(os.getcwd(), dataset_folder, data_filename)
    yield_file_path = os.path.join(os.getcwd(), dataset_folder, yield_filename)
    
    # Load image data and yield data
    image_data = np.load(data_file_path)
    yield_data = np.load(yield_file_path)
    
    return image_data, yield_data


def plot_history(history, save_path=None, dpi=300):
    """
    Plot the training history of a model and save it with high resolution.

    Parameters:
        history (History object): History object returned by model.fit().
        save_path (str, optional): File path to save the plot. If None, the plot is not saved.
        dpi (int, optional): Resolution (dots per inch) for the saved image.

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['mae'], label='mae')
    plt.plot(history.history['val_mae'], label='val_mae')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim([0, 40])  # Adjust the y-axis limits if necessary
    plt.legend(loc='upper right', fontsize=12)

    # Increase font size for ticks
    plt.xticks(fontsize=12)  # Adjust the font size as needed
    plt.yticks(fontsize=12)  # Adjust the font size as needed

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi)
        print(f"Plot saved at: {save_path}")

    plt.show()

def plot_history_2(history, save_path=None, dpi=300):
    """
    Plot the training history of a model and save it with high resolution.

    Parameters:
        history (History object): History object returned by model.fit().
        save_path (str, optional): File path to save the plot. If None, the plot is not saved.
        dpi (int, optional): Resolution (dots per inch) for the saved image.

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['mae'], label='mae')
    plt.plot(history.history['val_mae'], label='val_mae')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim([0, 0.5])  # Adjust the y-axis limits if necessary
    plt.legend(loc='upper right', fontsize=12)

    # Increase font size for ticks
    plt.xticks(fontsize=12)  # Adjust the font size as needed
    plt.yticks(fontsize=12)  # Adjust the font size as needed

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi)
        print(f"Plot saved at: {save_path}")

    plt.show()


def save_history(history, save_path):
    """
    Save the training history to a file using pickle.

    Parameters:
        history (dict): The training history object.
        save_path (str): File path to save the history.

    Returns:
        None
    """
    with open(save_path, 'wb') as f:
        pickle.dump(history, f)

def load_history(load_path):
    """
    Load the training history from a file using pickle.

    Parameters:
        load_path (str): File path to load the history from.

    Returns:
        dict: The loaded training history object.
    """
    with open(load_path, 'rb') as f:
        history = pickle.load(f)
    return history