import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, classes=None, title="Confusion Matrix"):
    """
    Plot a confusion matrix with counts in each cell.

    Parameters:
    - y_true: array-like of true labels
    - y_pred: array-like of predicted labels
    - labels: list of label names, optional
    - title: title of the plot
    """
    cm = confusion_matrix(y_true, y_pred,labels=classes)
    plt.figure(figsize=(9, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'plots/confusion_matrix_{title}.png', dpi=300)
    plt.show()

    

def plot_latent_space_2d(mus,labels,lr,num_epochs,only_quenched=False):
    
    'Plot the 2D latent space with quenched and unquenched data points.'

    plt.figure(figsize=(8,6))
    scatter_quenched = plt.scatter(mus[:, 0][labels==1], mus[:, 1][labels==1], c='red', alpha=0.7, label='Quenched')  
    scatter_unquenched = plt.scatter(mus[:, 0][labels==0], mus[:, 1][labels==0], c='blue', alpha=0.7,label='Unquenched') if not only_quenched else None
    plt.legend()
    plt.xlabel('Latent dim 1')
    plt.ylabel('Latent dim 2')
    plt.title('Spazio latente 2D')
    plt.grid(True)
    plt.text(0.05, 0.95, f'Learning Rate: {lr}, Epochs: {num_epochs}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    
    plt.savefig(f'plots/latent_space_2d_lr={lr}_{num_epochs}epochs.png', dpi=300)
    plt.show()
    
    return None