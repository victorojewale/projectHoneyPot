# visualizations/visualization.py

import matplotlib.pyplot as plt

def plot_accuracies(train_accuracies, val_accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
    #plt.savefig(save_path)
    plt.close()  
    #print(f"Plot saved to {save_path}")
