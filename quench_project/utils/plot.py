import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from train.trainer import Results
from train.trainer import Trainer
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred,
                          classes=None, labels=None,
                          title="Confusion Matrix", 
                          ax=None, fig_size=(9, 8)):
    """
    Plots a confusion matrix with counts in each cell.

    Parameters:
    - y_true: array-like of true labels.
    - y_pred: array-like of predicted labels.
    - classes: list of class values to index the confusion matrix (passed to sklearn).
    - labels: list of label names to show on axes (for display).
    - title: title of the plot.
    - ax: matplotlib Axes object. If None, a new figure and axes are created.
    - fig_size: size of the figure if ax is None.

    Returns:
    - fig: the matplotlib Figure object if created, otherwise None.
    """
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # Create a new figure if no axis is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = None
   
    # Plot the heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels if labels else classes,
                yticklabels=labels if labels else classes,
                ax=ax)

    # Label axes and set title
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

    return fig


    

def plot_latent_space_2d(mus,labels,lr,num_epochs,only_quenched=False,name='VAE'):
    
    'Plot the 2D latent space with quenched and unquenched data points.'

    fig,ax=plt.subplots(figsize=(8,6))
    scatter_quenched = plt.scatter(mus[:, 0][labels==1], mus[:, 1][labels==1], c='red', alpha=0.7, label='Quenched')  
    scatter_unquenched = plt.scatter(mus[:, 0][labels==0], mus[:, 1][labels==0], c='blue', alpha=0.7,label='Unquenched') if not only_quenched else None
    ax.legend()
    ax.set_xlabel('Latent dim 1')
    ax.set_ylabel('Latent dim 2')
    ax.set_title(f'Spazio latente 2D - {name}')
    ax.grid(True)
    ax.text(0.05, 0.95, f'Learning Rate: {lr}, Epochs: {num_epochs}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    
    
    
    return fig




def plot_results_from_training(results,trainer):
    
    train_losses = results.get_train_loss()
    val_losses = results.get_val_loss()

    train_losses_per_batch = results.get_train_losses_per_batch() 
    val_accuracy_per_epoch = results.get_val_accuracy() 
    val_recall_per_epoch = results.get_val_recall()

    num_epochs = trainer.num_epochs
    batch_size = trainer.batch_size
    lr = trainer.lr
    epoch_x_axis = np.arange(1, num_epochs + 1)
    
    fig= plt.figure(figsize=(16, 10))
    
    # Loss
    ax=fig.add_subplot(2, 2, 1)
    ax.plot(epoch_x_axis, train_losses, label='Avg Train Loss per Epoch', color='blue', linewidth=2)
    ax.plot(epoch_x_axis, val_losses, label='Val Loss per Epoch', color='red', linestyle='--', linewidth=2)

    if train_losses_per_batch is not None:

        num_batches_per_epoch = len(np.array(train_losses_per_batch)) // num_epochs
        batch_x_axis = np.linspace(1, num_epochs, len(np.array(train_losses_per_batch)))
        ax.plot(batch_x_axis, np.array(train_losses_per_batch), label='Train Loss per Batch', alpha=0.3, color='orange') 

    ax.text(0.05, 0.95, f'Learning Rate: {trainer.lr}, Epochs: {num_epochs} \nBatch size={batch_size}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ymax= max(max(np.array(train_losses)), max(np.array(val_losses))) * 1.1
    ymin = min(min(np.array(train_losses)), min(np.array(val_losses))) *0.9
    ax.set_ylim(ymin, ymax )  
    ax.set_title('Training and Test Losses')
    ax.legend()
    ax.grid(True)

    if val_accuracy_per_epoch is not None:
        # Accuracy
        
        ax=fig.add_subplot(2, 2, 2)
        ax.plot(epoch_x_axis, val_accuracy_per_epoch, label='Val Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Val Accuracy per Epoch')
        ax.legend()
        ax.grid(True)
    if val_recall_per_epoch is not None:
        # Recall
        ax=fig.add_subplot(2, 2, 3)
        ax.plot(epoch_x_axis, val_recall_per_epoch, label='Val Recall', color='green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Recall (%)')
        ax.set_title('Val Recall per Epoch')
        ax.legend()
        ax.grid(True)
    
    
    return fig

