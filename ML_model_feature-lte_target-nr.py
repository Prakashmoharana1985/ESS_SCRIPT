#prerequisite to run the following script#
#Download the get-pip.py file from https://bootstrap.pypa.io/get-pip.py.
#copy the file to your directory where u want to execute the get-pip.py file #cp /mnt/c/MISC/Tarun/get-pip.py .
#python3 get-pip.py # i used pip3 since python3 is installed but u can run pip if u have python2 is installed#
# This is needed in WSL/UBUNTU, if the package is not downloaded# else leave this step #echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf#
#Install the modules required to run ur script
#pip install pandas (in case python2)
#pip3 install pandas
#source /env/bin/activate
#Install the ML/AI modules for to train the model
#pip3 install pyparsing
#pip3 install sklearn
#pip3 install scikit-learn
#######################################################
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

print("This script requires EMCA LTE log with traces:")
print("ESSCTRL.19")
print("mtd peek -ta essCtrlCe -sig ESSCTRLLTECI_ESSAGENT_DL_FEEDBACK_IND INCOMING")
print("mtd peek -ta essCtrlCe -sig ESSCTRLCI_E5L_NR_FEEDBACK_IND")
#================================
# Get the filename to grep from the user
filename = input("Enter the filename to grep from: ")

# Apply grep command to extract lines containing "weightDl" or "WeightDl"
subprocess.run(["grep", "-E", "weightDl|WeightDl", filename], stdout=subprocess.PIPE, check=True)
#========================================

# Apply grep command to extract lines containing "weightDl" or "WeightDl"
#subprocess.run(["grep", "-E", "weightDl|WeightDl", filename], stdout=subprocess.PIPE, check=True)

# Save filtered data to "parsed.log"
with open("parsed.log", "w") as f2:
    f2.write(subprocess.run(["grep", "-E", "weightDl|WeightDl", filename], stdout=subprocess.PIPE, check=True).stdout.decode('utf-8'))

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
df = pd.read_csv('out3.csv', header=None, delimiter=';', names=['lte', 'nr'])

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
  # Plot the coefficients, intercept, and MSE
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    fig.suptitle('Linear Regression Results')

    axs[0].bar(df.columns[:-1], model.coef_)
    axs[0].set_title('Coefficients')
    axs[0].set_xlabel('Feature')
    axs[0].set_ylabel('Coefficient')

    axs[1].bar(['Intercept'], [model.intercept_])
    axs[1].set_title('Intercept')
    axs[1].set_xlabel('Metric')
    axs[1].set_ylabel('Value')

    axs[2].bar(['MSE'], [mse])
    axs[2].set_title('Mean Squared Error')
    axs[2].set_xlabel('Metric')
    axs[2].set_ylabel('Value')

    plt.show()