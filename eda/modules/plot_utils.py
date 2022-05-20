import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_feature_distribution(
    feature,
    fig, 
    sub_id = 111,
    bins   = 20,
    lw     = 3,
    ecolor = "white",
    color  = "#7eb0d5"
):
    
    ax = fig.add_subplot(sub_id)
    
    ax.hist(
        feature,
        bins      = bins,
        histtype  = "bar", 
        linewidth = lw,
        edgecolor = ecolor, 
        facecolor = color, 
        alpha     = 1, 
    )
    
    return ax


def plot_feature_correlation(
    x,
    y,
    fig, 
    sub_id = 111,
    color  = "#7eb0d5",
    size   = 20,
):

    ax = fig.add_subplot(sub_id)
    
    ax.scatter(
        x     = x,
        y     = y,
        color = color,
        s     = size
    )
    
    return ax




def plot_feature_correlation_hue(
    x,
    y,
    fig, 
    sub_id = 111,
    size   = 20,
    c      = None
):

    ax = fig.add_subplot(sub_id)
    
    ax.scatter(
        x     = x,
        y     = y,
        s     = size,
        c     = c
    )
    
    return ax