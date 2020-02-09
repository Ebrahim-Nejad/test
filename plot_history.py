# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 18:42:03 2020

@author: EBRAE
"""

def plot_history(net_history):
    history = net_history.history
    import matplotlib.pyplot as plt
    losses=history['loss']
    val_losses=history['val_loss']
    accuracies=history['acc']
    val_accuracies=history['val_acc']
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.plot(val_losses)
    plt.legend(['loss', 'val_loss'])
    
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(accuracies)
    plt.plot(val_accuracies)
    plt.legend(['accuracies', 'val_accuracies'])