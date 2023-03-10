"""
Some useful plotting functions.
"""


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("seaborn-deep")

from src.base import DEFAULT_FIG_SIZE, ArrayLike, HeightAndWidth


# =============================================================================
#  FUNCTIONS
# =============================================================================

def plot_feature_distributions(
    df: pd.DataFrame,
    bins: int = 50,
    figsize: HeightAndWidth = DEFAULT_FIG_SIZE,
) -> plt.figure:
    """ Plot histograms of continuous features and bar charts of discrete features.
    
    Args:
        df (pd.DataFrame) : Training DataFrame containing features and labels
        bins (int) : Number of bins for the continuous variable histograms
        figsize (HeightAndWidth) : Figure size

    Returns:
        fig (plt.figure) : The generated figure object
    """
    # Make sure the input is the correct DataFrame
    assert type(df) == pd.DataFrame
    assert all(col in df.columns for col in ["x", "y", "ft_home", "ft_away"])
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Plot histogram of x
    axes[0][0].hist(df["x"], bins=bins, facecolor="C0", edgecolor="C0", alpha=0.8)
    axes[0][0].set_title("x")

    # Plot histogram of y
    axes[0][1].hist(df["y"], bins=bins, facecolor="C1", edgecolor="C1", alpha=0.8)
    axes[0][1].set_title("y")

    # Plot bar chart of home scores
    home_value_counts = df["ft_home"].value_counts()
    home_scores = home_value_counts.index.astype(int).tolist()
    home_counts = home_value_counts.values.tolist()
    axes[1][0].bar(home_scores, home_counts, facecolor="C2", edgecolor="C2", alpha=0.8)
    axes[1][0].set_title("ft_home")

    # Plot bar chart of away scores
    away_value_counts = df["ft_away"].value_counts()
    away_scores = away_value_counts.index.astype(int).tolist()
    away_counts = away_value_counts.values.tolist()
    axes[1][1].bar(away_scores, away_counts, facecolor="C3", edgecolor="C3", alpha=0.8)
    axes[1][1].set_title("ft_away")

    return fig


def plot_targets_2D(df: pd.DataFrame, figsize: HeightAndWidth = DEFAULT_FIG_SIZE,):
    """ Plot 2D scatterplots of the targets ft_home and ft_away (0-5 goals).
    
    Args:
        df (pd.DataFrame) : Training DataFrame containing features and labels
        figsize (HeightAndWidth) : Figure size

    Returns:
        fig (plt.figure) : The generated figure object """

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for goals in range(6):
        mask = df["ft_home"].values == goals
        axes[0].scatter(df["x"].values[mask], df["y"].values[mask], 
            facecolor="C"+str(goals), edgecolor="w", s=20, alpha=0.9, label=goals)
    axes[0].set_title("ft_home")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend()

    for goals in range(6):
        mask = df["ft_away"].values == goals
        axes[1].scatter(df["x"].values[mask], df["y"].values[mask], 
            facecolor="C"+str(goals), edgecolor="w", s=20, alpha=0.9, label=goals)
    axes[1].set_title("ft_away")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].legend()

    return fig


def plot_elbos(
    elbos: list,
    title: str = "",
    colour: str = "C0",
    figsize: HeightAndWidth = DEFAULT_FIG_SIZE
) -> plt.figure:
    """ Plot ELBO vs. number of training iterations.
    
    Args:
        elbos (list) : List of ELBO values corresponding to each training iteration
        title (str) : Plot title
        colour (str) : Colour of the curve to plot
        figsize (HeightAndWidth) : Figure size

    Returns:
        fig (plt.figure) : The generated figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot ELBO's
    ax.plot(np.arange(len(elbos)), elbos, c=colour)
    ax.set_title(title)
    ax.set_xlabel("iteration")
    ax.set_ylabel("ELBO")
    
    return fig
