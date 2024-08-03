import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def readData(csv, county_name):
    countyData = csv[csv['county_name'] == county_name].copy()
    countyData['median_income'] = pd.to_numeric(countyData['median_income'])
    countyData = countyData.dropna(subset=['median_income'])
    countyData['median_income_display'] = countyData['median_income'].apply(lambda x: '${:,.2f}'.format(x))
    return countyData

def printDataSample(countyData):
    print("Sample data:")
    print(countyData[['racial_ethnic_group', 'year', 'median_income']].head(10))
    print("\nAvailable racial categories:")
    print(countyData['racial_ethnic_group'].unique())

def plotIndividualRegressions(countyData, county_name):
    countyData = countyData[countyData['racial_ethnic_group'] != "All"]
    printDataSample(countyData)
    racial_groups = countyData['racial_ethnic_group'].unique()
    
    for group in racial_groups:
        group_data = countyData[countyData['racial_ethnic_group'] == group]
        if group_data.empty:
            print(f"No data available for {group} in {county_name}.")
            continue
        X = group_data[['year']]
        y = group_data['median_income']
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        
        plt.figure(figsize=(12, 8))
        plt.plot(group_data['year'], y_pred, label=f'{group} Trend', linestyle='--')
        plt.scatter(group_data['year'], y, label=f'{group} Actual', alpha=0.7)
        plt.title(f'Linear Regression for {group} in {county_name}')
        plt.xlabel('Year')
        plt.ylabel('Median Income')
        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    csv = pd.read_csv('data.csv')
    phillyData = readData(csv, 'Philadelphia')
    print("Philadelphia County Data Loaded")
    plotIndividualRegressions(phillyData, 'Philadelphia')

if __name__ == "__main__":
    main()
