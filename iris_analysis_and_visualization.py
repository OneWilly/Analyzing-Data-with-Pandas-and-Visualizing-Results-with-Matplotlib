"""
Iris Dataset Analysis and Visualization
---------------------------------------
This script performs the following tasks:
1. Loads and explores the Iris dataset.
2. Computes basic data analysis statistics like mean, median, and standard deviation.
3. Creates various visualizations to explore relationships in the data.
4. Provides observations and insights after each visualization.

Requirements:
- pandas
- matplotlib
- seaborn
- scikit-learn
"""

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
def load_and_explore_data():
    """
    Load the Iris dataset, explore its structure, and check for missing values.
    """
    try:
        # Load the Iris dataset using sklearn
        iris = load_iris()
        
        # Convert the dataset into a pandas DataFrame
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = iris.target
        df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        
        # Display the first few rows of the dataset
        print("First 5 rows of the dataset:")
        print(df.head())
        
        # Check the structure of the dataset
        print("\nDataset Info:")
        print(df.info())
        
        # Check for missing values
        print("\nNumber of missing values in each column:")
        print(df.isnull().sum())
        
        # Since the Iris dataset has no missing values:
        print("\nNo missing values detected. Dataset is clean!")
        
        return df
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return None

# Task 2: Basic Data Analysis
def basic_data_analysis(df):
    """
    Perform basic data analysis on the dataset.
    """
    # Compute basic statistics for numerical columns
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Perform grouping by species and compute the mean of numerical columns
    grouped_data = df.groupby('species').mean()
    print("\nMean values of numerical columns grouped by species:")
    print(grouped_data)
    
    # Observations
    print("\nObservations:")
    print("1. Virginica species has the highest mean for petal length and petal width.")
    print("2. Setosa species has the smallest mean for all numerical columns.")

# Task 3: Data Visualization
def visualize_data(df):
    """
    Create visualizations to explore the dataset.
    """
    # Set a seaborn style for better aesthetics
    sns.set(style="whitegrid")
    
    # Line chart: Petal length trend for each species
    plt.figure(figsize=(8, 6))
    for species in df['species'].unique():
        subset = df[df['species'] == species]
        plt.plot(range(len(subset)), subset['petal length (cm)'], label=species)
    plt.title("Petal Length Trend by Species", fontsize=14)
    plt.xlabel("Index", fontsize=12)
    plt.ylabel("Petal Length (cm)", fontsize=12)
    plt.legend(title="Species", fontsize=10)
    plt.grid(alpha=0.5)
    plt.show()
    print("\nObservation: The line chart shows clear differences in petal length trends across species.")
    
    # Bar chart: Average petal width grouped by species
    plt.figure(figsize=(8, 6))
    sns.barplot(x='species', y='petal width (cm)', data=df, palette="viridis")
    plt.title("Average Petal Width by Species", fontsize=14)
    plt.xlabel("Species", fontsize=12)
    plt.ylabel("Petal Width (cm)", fontsize=12)
    plt.show()
    print("\nObservation: Virginica species has the largest average petal width.")
    
    # Histogram: Distribution of sepal length
    plt.figure(figsize=(8, 6))
    plt.hist(df['sepal length (cm)'], bins=15, color='skyblue', edgecolor='black')
    plt.title("Distribution of Sepal Length", fontsize=14)
    plt.xlabel("Sepal Length (cm)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(alpha=0.5)
    plt.show()
    print("\nObservation: Most sepal lengths are concentrated between 5 and 6 cm.")
    
    # Scatter plot: Sepal length vs. petal length
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df, palette="deep")
    plt.title("Sepal Length vs. Petal Length", fontsize=14)
    plt.xlabel("Sepal Length (cm)", fontsize=12)
    plt.ylabel("Petal Length (cm)", fontsize=12)
    plt.legend(title="Species", fontsize=10)
    plt.grid(alpha=0.5)
    plt.show()
    print("\nObservation: There is a strong positive correlation between petal length and sepal length.")

    # Additional visualization: Pairplot for pairwise relationships
    sns.pairplot(df, hue='species', palette='viridis', diag_kind='hist', height=2.5)
    plt.suptitle("Pairwise Relationships Among Features", y=1.02, fontsize=16)
    plt.show()
    print("\nObservation: Pairplot shows clear separations among species in petal-related dimensions.")

# Main function to execute the tasks
def main():
    """
    Main function to execute all tasks in the assignment.
    """
    print("Loading and exploring the dataset...")
    df = load_and_explore_data()
    if df is not None:
        print("\nPerforming basic data analysis...")
        basic_data_analysis(df)
        print("\nCreating visualizations...")
        visualize_data(df)
        print("\nAssignment tasks completed successfully!")

# Execute the script
if __name__ == "__main__":
    main()
