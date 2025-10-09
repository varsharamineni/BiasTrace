import random
import pandas as pd

# Set random seed for reproducibility
random.seed(42)

# Define labellers and data points
labellers = ['VR', 'KS', 'JR', 'SR']
data_points = list(range(100))

# Create a list with 25 of each labeller
labeller_assignments = labellers * 25

# Shuffle the labeller assignments randomly
random.shuffle(labeller_assignments)

# Assign each data point to a labeller
assignments = [{'DataPoint': dp, 'Labeller': lb} 
               for dp, lb in zip(data_points, labeller_assignments)]

# Create DataFrame
df = pd.DataFrame(assignments)

# Display table
print(df)

# Optionally, save to CSV
df.to_csv('reasoning_error_taxonomy/assignments.csv', index=False)