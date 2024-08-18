import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

def readData(csv, county_name):
    countyData = csv[csv['county_name'] == county_name].copy()
    countyData['median_income'] = pd.to_numeric(countyData['median_income'])
    countyData = countyData.dropna(subset=['median_income'])
    return countyData

def decisionTreeRegression(countyData, county_name):
    filtered_data = countyData[countyData['racial_ethnic_group'].isin(['White alone, non-Latinx', 'Black or African American alone'])]
    plt.figure(figsize=(12, 8))
    for group, color in zip(['White alone, non-Latinx', 'Black or African American alone'], ['blue', 'green']):
        group_data = filtered_data[filtered_data['racial_ethnic_group'] == group]
        X = group_data[['year']]
        y = group_data['median_income']
        
        tree_model = DecisionTreeRegressor().fit(X, y)
        y_pred = tree_model.predict(X)
        
        plt.plot(group_data['year'], y_pred, label=f'{group} Trend', linestyle='--', color=color)
        plt.scatter(group_data['year'], y, label=f'{group} Actual', alpha=0.7, color=color)
    
    plt.title(f'Decision Tree Regression for {county_name} (Each Year as Category)')
    plt.xlabel('Year')
    plt.ylabel('Median Income')
    plt.legend()
    plt.grid(True)
    plt.show()

    filtered_data['year_group'] = pd.cut(filtered_data['year'], bins=np.arange(2000, 2028, 5), right=False, labels=[f'{year}-{year+4}' for year in range(2000, 2025, 5)])
    grouped_data = filtered_data.groupby(['racial_ethnic_group', 'year_group']).agg({'median_income': 'mean'}).reset_index()
    
    plt.figure(figsize=(12, 8))
    for group, color in zip(['White alone, non-Latinx', 'Black or African American alone'], ['red', 'purple']):
        group_data = grouped_data[grouped_data['racial_ethnic_group'] == group]
        X = group_data[['year_group']].apply(lambda x: x.cat.codes).values.reshape(-1, 1)
        y = group_data['median_income']
        
        tree_model = DecisionTreeRegressor().fit(X, y)
        y_pred = tree_model.predict(X)
        
        plt.plot(group_data['year_group'].cat.codes, y_pred, label=f'{group} Trend', linestyle='--', color=color)
        plt.scatter(group_data['year_group'].cat.codes, y, label=f'{group} Actual', alpha=0.7, color=color)
    
    plt.title(f'Decision Tree Regression for {county_name} (5-Year Increments)')
    plt.xlabel('5-Year Group')
    plt.ylabel('Median Income')
    plt.xticks(ticks=range(len(grouped_data['year_group'].unique())), labels=grouped_data['year_group'].unique())
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    csv = pd.read_csv('data.csv')
    phillyData = readData(csv, 'Philadelphia')
    print("Philadelphia County Data Loaded")
    decisionTreeRegression(phillyData, 'Philadelphia')

if __name__ == "__main__":
    main()

