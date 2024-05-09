from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np

# Load MNIST data from OpenML
mnist = fetch_openml('mnist_784', version=1)

# Convert to DataFrame
full_data = pd.DataFrame(data=mnist.data, columns=[f'pixel{i}' for i in range(784)])
full_data['label'] = mnist.target

# Sample 10% of the data
sampled_data = full_data.sample(frac=0.1, random_state=42)  # Random state for reproducibility

file_path = '/Users/michalkielkowski/Desktop/infa-all/magisterka/wizualizacja_duzych_zbiorów_danych/Projekt/WDZD_Project/data/mnist_dataset_sampled.csv'

# Save to CSV
sampled_data.to_csv(file_path, index=False)
print("end\n")



