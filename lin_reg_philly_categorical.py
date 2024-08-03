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

def linearRegressionWithCategories(countyData, county_name):

    countyData = countyData[countyData['racial_ethnic_group'] != "All Races"]
    

    X_year = countyData['year'].values.reshape(-1, 1)
    X_racial = countyData['racial_ethnic_group'].values.reshape(-1, 1)
    y = countyData['median_income'].values

    encoder = OneHotEncoder(drop='first')
    X_racial_encoded = encoder.fit_transform(X_racial).toarray()

    X_combined = np.hstack((X_year, X_racial_encoded))
    
    model = LinearRegression().fit(X_combined, y)
    y_pred = model.predict(X_combined)
    plt.figure(figsize=(20, 6))
    unique_racial_groups = countyData['racial_ethnic_group'].unique()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_racial_groups)))
    
    for color, ethnic_group in zip(colors, unique_racial_groups):
        group_data = countyData[countyData['racial_ethnic_group'] == ethnic_group]
        X_group = group_data['year'].values
        y_group = group_data['median_income'].values
        y_pred_group = model.predict(np.hstack((X_group.reshape(-1, 1), encoder.transform(group_data[['racial_ethnic_group']]).toarray())))
        
        plt.scatter(X_group, y_group, label=f'{ethnic_group} (actual)', alpha=0.7, color=color)
        plt.plot(X_group, y_pred_group, label=f'{ethnic_group} (predicted)', linestyle='--', color=color)

    plt.title(f'Combined Linear Regression of Median Income for All Ethnic Groups in {county_name} Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Median Income')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    csv = pd.read_csv('data.csv')
    
    phillyData = readData(csv, 'Philadelphia')
    print("Philadelphia County Data Loaded")
    linearRegressionWithCategories(phillyData, 'Philadelphia')

if __name__ == "__main__":
    main()



