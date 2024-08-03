import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

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
    
    summaries = []
    
    for group in racial_groups:
        group_data = countyData[countyData['racial_ethnic_group'] == group]
        if group_data.empty:
            print(f"No data available for {group} in {county_name}.")
            continue
        X = group_data[['year']]
        y = group_data['median_income']
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        X_with_const = sm.add_constant(X)
        ols_model = sm.OLS(y, X_with_const).fit()
        slope = model.coef_[0]
        intercept = model.intercept_
        p_value = ols_model.pvalues[1]
        f_stat = ols_model.fvalue
        df_model = int(ols_model.df_model)
        df_resid = int(ols_model.df_resid)
        summary = (f"{group}: Median income = {slope:.3f} * Year + {intercept:.2f}, "
                   f"R^2 = {r2:.3f}, F({df_model},{df_resid}) = {f_stat:.2f}, p = {p_value:.3f}.")
        summaries.append(summary)
     
        plt.figure(figsize=(12, 8))
        plt.plot(group_data['year'], y_pred, label=f'{group} Trend', linestyle='--')
        plt.scatter(group_data['year'], y, label=f'{group} Actual', alpha=0.7)
        plt.title(f'Linear Regression for {group} in {county_name}')
        plt.xlabel('Year')
        plt.ylabel('Median Income')
        plt.legend()
        plt.grid(True)
        plt.show()

    print("\nRegression Summary:")
    for i, summary in enumerate(summaries, start=1):
        print(f"{i}. {summary}")

def main():
    csv = pd.read_csv('data.csv')
    phillyData = readData(csv, 'Philadelphia')
    print("Philadelphia County Data Loaded")
    plotIndividualRegressions(phillyData, 'Philadelphia')

if __name__ == "__main__":
    main()
