import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS, results
import statsmodels.api as sm

data = pd.read_excel(r'E:/Users/A2010290/Documents/GitHub/NoShort/data/US_state_data.xlsx', sheet_name=None, index_col=0)

sheet_names_data = list(data.keys())[2:]
US_rates = data['US rate']
annual_returns = data['Returns']
for j, sheet_name_data in enumerate(sheet_names_data):
    parti_rates = pd.DataFrame.copy(data[sheet_name_data])
    parti_rates = parti_rates.drop(index={'United States', 'Other Areas'})

    # detrend_national = pd.DataFrame(
    #     parti_rates_1.values - parti_rates_1.iloc[[-1]].values,
    #     index=parti_rates_1.index,
    #     columns=parti_rates_1.columns
    # )
    # detrend_national = detrend_national.drop(index='United States')
    detrend_state = pd.DataFrame.copy(parti_rates)
    parti_rates_fit = pd.DataFrame.copy(parti_rates)
    year = parti_rates_fit.columns
    detrend_state.columns = year

    parti_rates_fit.loc['year_index'] = year - year[0]
    states = parti_rates.index
    x1 = parti_rates_fit.loc['year_index']
    x = sm.add_constant(x1)
    for i, state in enumerate(states):  # detrend the data
        y = parti_rates_fit.loc[state]
        lm = sm.OLS(y, x).fit()
        lm_fit = lm.predict(x)
        # print(lm.params[1])
        detrend_state.loc[state] = y - lm_fit

    # a = parti_rates_1.melt(ignore_index=False).reset_index()
    # a = detrend_national.melt(ignore_index=False).reset_index()
    if j == 0:  # merge the returns to the participation rate panel
        a = detrend_state.melt(ignore_index=False).reset_index()
        a = a.rename(columns={'variable': 'year', 'value': 'parti rate' + sheet_name_data})

    else:
        c = detrend_state.melt(ignore_index=False).reset_index()
        c = c.rename(columns={'variable': 'year', 'value': 'parti rate' + sheet_name_data})
        a = pd.merge(a, c)

col_names = annual_returns.index

for i, col_name in enumerate(col_names):
    b = (annual_returns.iloc[[i]]).melt(ignore_index=False).reset_index()  # each row to panel
    b = b.drop(columns=[b.columns[0]])
    b = b.rename(columns={'variable': 'year', 'value': col_name})
    # b['year'] = b['year']  # returns lagged one year
    b['year'] = b['year'] - 1  # returns lagged one year
    a = pd.merge(a, b, on='year')

a['Equity risk premium'] = a['S&P 500 (includes dividends)'] - a['3-month T.Bill']
a['Real S&P'] = a['S&P 500 (includes dividends)'] - a['Inflation Rate']
year = pd.Categorical(a.year)
a = a.set_index(['States', 'year'])
a['year'] = year
y_names = ['S&P 500 (includes dividends)', 'Equity risk premium', 'Real S&P']
for i in range(3):
    # exog_vars = [y_names[i], 'Gold*', ' Baa Corporate Bond', 'Real Estate']
    # exog_vars = [y_names[i], 'Real Estate']
    exog_vars = [y_names[i]]
    table = {
        '(1)': PanelOLS(dependent=a['parti rate' + sheet_names_data[0]],
                        exog=sm.add_constant(a[exog_vars]), entity_effects=True).fit(
            cov_type='robust',
            # cov_type='clustered', cluster_entity=True,
            # cluster_time=True
        ),
        '(2)': PanelOLS(dependent=a['parti rate' + sheet_names_data[1]],
                        exog=sm.add_constant(a[exog_vars]), entity_effects=True).fit(
            cov_type='robust',
            # cov_type='clustered', cluster_entity=True,
            # cluster_time=True
        ),
        '(3)': PanelOLS(dependent=a['parti rate' + sheet_names_data[2]],
                        exog=sm.add_constant(a[exog_vars]), entity_effects=True).fit(
            cov_type='robust',
            # cov_type='clustered', cluster_entity=True,
            # cluster_time=True
        ),
        '(4)': PanelOLS(dependent=a['parti rate' + sheet_names_data[3]],
                        exog=sm.add_constant(a[exog_vars]), entity_effects=True).fit(
            cov_type='robust',
            # cov_type='clustered', cluster_entity=True,
            # cluster_time=True
        ),
        '(5)': PanelOLS(dependent=a['parti rate' + sheet_names_data[4]],
                        exog=sm.add_constant(a[exog_vars]), entity_effects=True).fit(
            cov_type='robust',
            # cov_type='clustered', cluster_entity=True,
            # cluster_time=True
        ),
    }
    # display(results.compare(table))
    comparrison = results.compare(table)
    summary = comparrison.summary
    print(summary.as_latex())


