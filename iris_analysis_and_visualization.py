# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
def load_and_explore_data():
    """
    Load the Iris dataset, explore its structure, and clean the data if necessary.
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
        
        # Check for null values and dataset info
        print("\nDataset Info:")
        print(df.info())
        print("\nNumber of missing values in each column:")
        print(df.isnull().sum())
        
        # Since Iris dataset has no missing values, no cleaning is required
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
    print("\nObservation: Virginica species has the highest mean for petal length and width.")
    
# Task 3: Data Visualization
def visualize_data(df):
    """
    Create visualizations to explore the dataset.
    """
    # Set a seaborn style for better aesthetics
    sns.set(style="whitegrid")
    
    # Line chart: Trend of petal length for each species
    plt.figure(figsize=(8, 6))
    for species in df['species'].unique():
        subset = df[df['species'] == species]
        plt.plot(range(len(subset)), subset['petal length (cm)'], label=species)
    plt.title("Petal Length Trend by Species")
    plt.xlabel("Index")
    plt.ylabel("Petal Length (cm)")
    plt.legend()
    plt.grid()
    plt.show()
    
    # Bar chart: Average petal width grouped by species
    plt.figure(figsize=(8, 6))
    sns.barplot(x='species', y='petal width (cm)', data=df, palette="viridis")
    plt.title("Average Petal Width by Species")
    plt.xlabel("Species")
    plt.ylabel("Petal Width (cm)")
    plt.show()
    
    # Histogram: Distribution of sepal length
    plt.figure(figsize=(8, 6))
    plt.hist(df['sepal length (cm)'], bins=15, color='skyblue', edgecolor='black')
    plt.title("Distribution of Sepal Length")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Frequency")
    plt.show()
    
    # Scatter plot: Sepal length vs. petal length
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df, palette="deep")
    plt.title("Sepal Length vs. Petal Length")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Petal Length (cm)")
    plt.legend()
    plt.show()

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
