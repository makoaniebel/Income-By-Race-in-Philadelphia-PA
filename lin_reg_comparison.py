import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

def readData(csv, county_name):
    countyData = csv[csv['county_name'] == county_name].copy()
    countyData.loc[:, 'median_income'] = pd.to_numeric(countyData['median_income'])
    countyData = countyData.dropna(subset=['median_income'])
    countyData.loc[:, 'median_income_display'] = countyData['median_income'].apply(lambda x: '${:,.2f}'.format(x))
    return countyData

def plotComparisonGraph(countyData, county_name, group1, group2):
    print(f"Filtering data for {group1} and {group2}")
    countyData = countyData[countyData['racial_ethnic_group'].isin([group1, group2])]
    
    if countyData.empty:
        print(f"No data available for {group1} and {group2} in {county_name}.")
        return
    
    X_year = countyData['year'].values.reshape(-1, 1)
    X_racial = countyData['racial_ethnic_group'].values.reshape(-1, 1)
    y = countyData['median_income'].values
    
    encoder = OneHotEncoder(drop='first')
    
    try:
        X_racial_encoded = encoder.fit_transform(X_racial).toarray()
    except ValueError as e:
        print(f"Error encoding racial groups: {e}")
        return
    
    X_combined = np.hstack((X_year, X_racial_encoded))
    
    model = LinearRegression().fit(X_combined, y)
    
    plt.figure(figsize=(10, 6))
    
    for group in [group1, group2]:
        group_data = countyData[countyData['racial_ethnic_group'] == group]
        if group_data.empty:
            continue
        
        X_group = group_data['year'].values
        y_group = group_data['median_income'].values
        X_group_encoded = encoder.transform(group_data[['racial_ethnic_group']]).toarray()
        y_pred_group = model.predict(np.hstack((X_group.reshape(-1, 1), X_group_encoded)))
        
        plt.scatter(X_group, y_group, label=f'{group} (actual)', alpha=0.7)
        plt.plot(X_group, y_pred_group, label=f'{group} (predicted)', linestyle='--')
    
    plt.title(f'Comparison of Median Income Linear Regression for {group1} and {group2} in {county_name}')
    plt.xlabel('Year')
    plt.ylabel('Median Income')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    csv = pd.read_csv('data.csv')
    
    phillyData = readData(csv, 'Philadelphia')
    print("Philadelphia County Data Loaded")
    plotComparisonGraph(phillyData, 'Philadelphia', 'White alone, non-Latinx', 'Black or African American alone')
    plotComparisonGraph(phillyData, 'Philadelphia', 'Asian alone', 'Latinx')
    plotComparisonGraph(phillyData, 'Philadelphia', 'White alone, non-Latinx', 'People of Color')

if __name__ == "__main__":
    main()

