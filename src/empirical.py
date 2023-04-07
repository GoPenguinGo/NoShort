import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import statsmodels.api as sm

data = pd.read_excel(r'E:/Users/A2010290/Documents/GitHub/NoShort/data/US_state_data.xlsx', sheet_name=None, index_col=0)
US_rates = data['US rate']
annual_returns = data['Returns']
parti_rates_all = data['Rate by state  all']
parti_rates_50 = data['Rate by state  <50k']
parti_rates_50_100 = data['Rate by state  50k-100k']
parti_rates_100_200 = data['Rate by state  100k-200k']
parti_rates_200 = data['Rate by state  >200k']

parti_rates = pd.DataFrame.copy(parti_rates_200)
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
for i, state in enumerate(states):
    y = parti_rates_fit.loc[state]
    lm = sm.OLS(y, x).fit()
    lm_fit = lm.predict(x)
    print(lm.params[1])
    detrend_state.loc[state] = y - lm_fit

# a = parti_rates_1.melt(ignore_index=False).reset_index()
# a = detrend_national.melt(ignore_index=False).reset_index()
a = detrend_state.melt(ignore_index=False).reset_index()
a = a.rename(columns={'variable':'year', 'value':'parti rate'})
col_names = annual_returns.index
for i, col_name in enumerate(col_names):
    b = (annual_returns.iloc[[i]]).melt(ignore_index=False).reset_index()  # each row to panel
    b = b.drop(columns = [b.columns[0]])
    b = b.rename(columns={'variable':'year', 'value':col_name})
    b['year'] = b['year'] - 1  # returns lagged one year
    a = pd.merge(a, b, on='year')
year = pd.Categorical(a.year)
a = a.set_index(['States', 'year'])
a['year'] = year
m = PanelOLS(
    dependent=a['parti rate'],
    exog=a['S&P 500 (includes dividends)'],
    entity_effects=True,
    time_effects=False,
    )
m.fit(cov_type='clustered', cluster_entity=True, cluster_time=False)