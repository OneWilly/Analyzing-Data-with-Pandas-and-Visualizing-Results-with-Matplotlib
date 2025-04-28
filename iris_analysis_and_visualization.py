"""
Iris Dataset Analysis and Visualization (Refined)
----------------------------------------------
This script performs the following tasks:
1.  Loads and explores the Iris dataset.
2.  Computes basic data analysis statistics.
3.  Creates various visualizations.
4.  Provides observations and insights.

Improvements:
-   Parameterized data loading for flexibility.
-   Abstraction of visualization logic into helper functions.
-   Consistent use of f-strings for print statements.

Requirements:
-   pandas
-   matplotlib
-   seaborn
-   scikit-learn
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Configuration: Dataset file path (can be changed)
DATASET_FILE = "iris_dataset.csv"


def save_iris_to_csv(filepath="iris_dataset.csv"):
    """
    Saves the Iris dataset to a CSV file.

    Args:
        filepath (str): The path to save the CSV file.
    """
    try:
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df["species"] = iris.target
        df["species"] = df["species"].map(
            {0: "setosa", 1: "versicolor", 2: "virginica"}
        )
        df.to_csv(filepath, index=False)
        print(f"Iris dataset saved to '{filepath}'.")
    except Exception as e:
        print(f"An error occurred while saving the dataset: {e}")


def load_and_explore_data(filepath):
    """
    Loads and explores data from a CSV file.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame, or None if an error occurs.
    """
    try:
        df = pd.read_csv(filepath)

        print("\nFirst 5 rows of the dataset:")
        print(df.head())

        print("\nDataset Info:")
        print(df.info())

        print("\nNumber of missing values in each column:")
        print(df.isnull().sum())

        if df.isnull().sum().sum() == 0:
            print("\nNo missing values detected. Dataset is clean!")
        else:
            print("\nMissing values found. Handling not implemented in this example.")
            #   In a real scenario, you might fill or drop NaNs here

        return df
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return None


def basic_data_analysis(df):
    """
    Performs basic data analysis on the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
    """
    print("\nBasic Statistics:")
    print(df.describe())

    grouped_data = df.groupby("species").mean()
    print("\nMean values of numerical columns grouped by species:")
    print(grouped_data)

    print(
        "\nObservations:\n"
        "1. Virginica species has the highest mean for petal length and petal width.\n"
        "2. Setosa species has the smallest mean for all numerical columns."
    )


# --- Visualization Helper Functions ---
def create_line_chart(df, x_col, y_col, hue_col, title, xlabel, ylabel):
    """
    Creates a line chart.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        x_col (str): Column name for x-axis.
        y_col (str): Column name for y-axis.
        hue_col (str): Column name for grouping lines.
        title (str): Title of the plot.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
    """

    plt.figure(figsize=(8, 6))
    for hue_value in df[hue_col].unique():
        subset = df[df[hue_col] == hue_value]
        plt.plot(
            range(len(subset)), subset[y_col], label=hue_value
        )  # Using index as x
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(title=hue_col, fontsize=10)
    plt.grid(alpha=0.5)
    plt.show()


def create_bar_chart(df, x_col, y_col, title, xlabel, ylabel):
    """
    Creates a bar chart.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        x_col (str): Column name for x-axis.
        y_col (str): Column name for y-axis.
        title (str): Title of the plot.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
    """
    plt.figure(figsize=(8, 6))
    sns.barplot(x=x_col, y=y_col, data=df, palette="viridis")
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.show()


def create_histogram(df, x_col, title, xlabel, ylabel):
    """
    Creates a histogram.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        x_col (str): Column name for x-axis.
        title (str): Title of the plot.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
    """

    plt.figure(figsize=(8, 6))
    plt.hist(df[x_col], bins=15, color="skyblue", edgecolor="black")
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(alpha=0.5)
    plt.show()


def create_scatter_plot(df, x_col, y_col, hue_col, title, xlabel, ylabel):
    """
    Creates a scatter plot.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        x_col (str): Column name for x-axis.
        y_col (str): Column name for y-axis.
        hue_col (str): Column name for color-coding points.
        title (str): Title of the plot.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=x_col, y=y_col, hue=hue_col, data=df, palette="deep"
    )
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(title=hue_col, fontsize=10)
    plt.grid(alpha=0.5)
    plt.show()


def visualize_data(df):
    """
    Creates visualizations to explore the dataset.

    Args:
        df (pd.DataFrame): The DataFrame to visualize.
    """

    sns.set(style="whitegrid")  # Set seaborn style

    #   Line Chart
    create_line_chart(
        df,
        "Index",
        "petal length (cm)",
        "species",
        "Petal Length Trend by Species",
        "Index",
        "Petal Length (cm)",
    )
    print(
        "\nObservation: The line chart shows clear differences in petal "
        "length trends across species."
    )

    #   Bar Chart
    create_bar_chart(
        df,
        "species",
        "petal width (cm)",
        "Average Petal Width by Species",
        "Species",
        "Petal Width (cm)",
    )
    print("\nObservation: Virginica species has the largest average petal width.")

    #   Histogram
    create_histogram(
        df,
        "sepal length (cm)",
        "Distribution of Sepal Length",
        "Sepal Length (cm)",
        "Frequency",
    )
    print("\nObservation: Most sepal lengths are concentrated between 5 and 6 cm.")

    #   Scatter Plot
    create_scatter_plot(
        df,
        "sepal length (cm)",
        "petal length (cm)",
        "species",
        "Sepal Length vs. Petal Length",
        "Sepal Length (cm)",
        "Petal Length (cm)",
    )
    print(
        "\nObservation: There is a strong positive correlation between petal "
        "length and sepal length."
    )

    #   Pairplot (Additional)
    sns.pairplot(df, hue="species", palette="viridis", diag_kind="hist", height=2.5)
    plt.suptitle("Pairwise Relationships Among Features", y=1.02, fontsize=16)
    plt.show()
    print(
        "\nObservation: Pairplot shows clear separations among species in "
        "petal-related dimensions."
    )


def main():
    """
    Main function to execute all tasks.
    """

    print("Saving the Iris dataset to a CSV file...")
    save_iris_to_csv(DATASET_FILE)  #   Use the global constant

    print("\nLoading and exploring the dataset from the CSV file...")
    df = load_and_explore_data(DATASET_FILE)  #   Use the global constant
    if df is not None:
        print("\nPerforming basic data analysis...")
        basic_data_analysis(df)
        print("\nCreating visualizations...")
        visualize_data(df)
        print("\nAssignment tasks completed successfully!")


if __name__ == "__main__":
    main()
