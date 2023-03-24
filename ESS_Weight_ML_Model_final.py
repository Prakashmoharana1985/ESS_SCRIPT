import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sys

print("This script requires EMCA LTE log with traces:")
print("ESSCTRL.19")
print("mtd peek -ta essCtrlCe -sig ESSCTRLLTECI_ESSAGENT_DL_FEEDBACK_IND INCOMING")
print("mtd peek -ta essCtrlCe -sig ESSCTRLCI_E5L_NR_FEEDBACK_IND")
#================================
# Get the filename to grep from standard input
filename = sys.stdin.read().encode('utf-8')

# Apply grep command to extract lines containing "weightDl" or "WeightDl"
grep_output = subprocess.run(["grep", "-E", "weightDl|WeightDl"], input=filename, stdout=subprocess.PIPE, check=True)

# Save filtered data to "parsed.log"
with open("parsed.log", "w") as f2:
    f2.write(grep_output.stdout.decode('utf-8'))

INPUT_FILE = "parsed.log"
OUTPUT_FILE = "out3.csv"

print("INPUT file = " + INPUT_FILE + ", this should contain your traces")
print("OUTPUT file = " + OUTPUT_FILE + ", drag and drop to excel to plot")

# Parsed log file
f = open(INPUT_FILE)
# Output file
f2 = open(OUTPUT_FILE,"w")

lteWeight = 0

for line in f:
    # extracting the LTE DL weight
    if "lteRpBandWeightDl: " in line:
        lteWeight = int(line[22:])
    # extracting this NR DL weight
    if "} weightDl=" in line:
        start = line.find("weightDl=") +9
        end = line.find("weightUl=") -1
        nrWeight = int(line[start:end])
        f2.write(str(lteWeight) + " ; " + str(nrWeight) +"\n")

# Close the output file
f2.close()

# Load the data into a pandas dataframe and split into 'lte' and 'nr' columns
df = pd.read_csv(OUTPUT_FILE, header=None, delimiter=';', names=['lte', 'nr'])

# Check if the dataframe is empty
if df.empty:
    print("Error: no data in the CSV file")
else:
    # Create a line plot of lte_weight in blue color
    plt.plot(df.index, df['lte'], color='blue', label='lte_weight')

    # Create a line plot of nr_weight in red color
    plt.plot(df.index, df['nr'], color='red', label='nr_weight')

    # Set the labels for the axes
    plt.xlabel('instances')
    plt.ylabel('weight')

    # Add a legend to the plot
    plt.legend()

    # Show the plot
    plt.show()
    # Summary statistics of the data
    summary_stats = df.describe()
    print(summary_stats)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df[['lte']], df['nr'], test_size=0.2, random_state=42)

    # Train a linear regression model on the training data
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Print the coefficients and intercept of the model
    print('Coefficients:', model.coef_)
    print('Intercept:', model.intercept_)

    # Make predictions on the test data and print the mean squared error
    y_pred = model.predict(X_test)
    mse = ((y_pred - y_test) ** 2).mean()
    print('Mean Squared Error:', mse)