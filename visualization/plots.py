"""Plots used by the Streamlit interface."""

import matplotlib.pyplot as plt
import numpy as np


def variance_plot(ratios, cumulative):
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(1, len(ratios) + 1)
    ax.bar(x, ratios, alpha=0.45, label="Individual explained variance")
    ax.plot(x, cumulative, marker="o", markersize=3, label="Cumulative explained variance")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance ratio")
    ax.set_title("PCA Explained Variance")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig


def reconstruction_plot(ks, errors):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ks, errors, marker="o")
    ax.set_xlabel("Number of components (k)")
    ax.set_ylabel("Mean squared error")
    ax.set_title("Reconstruction Error vs Components")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def scatter_plot(x, y, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, s=18, alpha=0.65)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Observations in Principal-Component Space")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig
