import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS, results
import statsmodels.api as sm
import os
from sklearn.linear_model import LinearRegression

data = pd.read_excel(r'E:/Users/A2010290/Documents/GitHub/NoShort/data/US_state_data.xlsx', sheet_name=None,
                     index_col=0)
match_names = pd.read_excel(r'E:/Users/A2010290/Documents/GitHub/NoShort/data/State_names.xlsx')
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

# # Zip code level data:
# path = r'E:\Users\A2010290\Documents\GitHub\NoShort\data\Zip code data'
# data_keep = ['STATE', 'ZIPCODE', 'AGI_CLASS', 'N1', 'N00600']
# merged_data_numbers = []
# merged_data_parti_rate = []
# for filename in os.listdir(path):
#     if filename.endswith('.csv'):
#         print(filename)
#         data_read = pd.read_csv(os.path.join(path, filename))
#         year = int('20' + filename[:2])
#         data_focus = data_read[data_keep]
#         data_focus['STATE'] = data_focus['STATE'].str.upper()
#         zipcode_int = data_focus[data_keep[1]].astype('Int64')
#         data_focus.insert(0, 'year', year)
#         data_focus.insert(0, 'id', data_focus['STATE'] + '_' + zipcode_int.astype(str).str[:5] + '_' + data_focus[
#             data_keep[2]].astype(str))
#         data_focus.insert(0, 'parti_rate', data_focus['N00600'].div(data_focus['N1']))
#         data_focus = data_focus.loc[
#             (zipcode_int != 0) & (data_focus[data_keep[2]].astype(str) != '0') & (data_focus['N1'] != 0)]
#         data_focus_numbers = data_focus[['id', 'year', 'N00600', 'N1']]
#         data_focus_parti_rate = data_focus[['id', 'year', 'parti_rate']]
#         if year == 2005:
#             merged_data_numbers = data_focus_numbers
#             merged_data_parti_rate = data_focus_parti_rate
#         else:
#             merged_data_numbers = merged_data_numbers.append(data_focus_numbers, ignore_index=True)
#             merged_data_parti_rate = merged_data_parti_rate.append(data_focus_parti_rate, ignore_index=True)
#
# # add state_year, and state_agi variables
# # if zip_code == 0: state overall
# merged_data_focus = pd.DataFrame.copy(merged_data_numbers)
# merged_data_focus.insert(0, 'parti_rate', merged_data_numbers['N00600'] / merged_data_numbers['N1'])
# merged_data_focus.insert(0, 'income_group', (merged_data_numbers['id'].astype(str).str[-1]).astype('category'))
# merged_data_focus = merged_data_focus.loc[
#     (merged_data_focus['income_group'] != '0') & (merged_data_focus['parti_rate'] > 0) & (
#             merged_data_focus['parti_rate'] < 1) & (merged_data_focus['parti_rate'] != np.inf)]
# merged_data_focus.insert(0, 'state', (merged_data_numbers['id'].astype(str).str[:2]).astype('category'))
# merged_data_focus.insert(0, 'state_income', (
#         merged_data_numbers['id'].astype(str).str[:2] + merged_data_numbers['id'].astype(str).str[-1]).astype(
#     'category'))
# merged_data_focus.insert(0, 'state_zip', (merged_data_numbers['id'].astype(str).str[:-2]).astype('category'))
# # # Report the frequency that a value occurs in a column:
# # df.groupby('a').count()
#
# # detrend in each state_income group
# merged_data_fit = pd.DataFrame.copy(merged_data_focus)
# merged_data_fit.insert(0, 'parti_rate_detrend', merged_data_fit.parti_rate)
# years = merged_data_numbers['year'].unique()
# merged_data_fit = merged_data_fit[merged_data_fit.groupby('id')['id'].transform('count').ge(10)]
# state_income_groups = merged_data_fit.state_income.unique()
# for i, state_income_group in enumerate(state_income_groups):
#     df1 = merged_data_fit[(merged_data_fit['state_income'] == state_income_group)]
#     x = sm.add_constant(df1.year)
#     y = df1.parti_rate
#     lm = sm.OLS(y, x).fit()
#     lm_fit = lm.predict(x)
#     df1['parti_residual'] = y - lm_fit
#     merged_data_fit.loc[merged_data_fit.index.isin(df1.index), 'parti_rate_detrend'] = df1.parti_residual
#
# merged_data_returns = pd.DataFrame.copy(merged_data_fit)
# # merge with the returns data
#
# annual_returns_detrend = pd.DataFrame.copy(annual_returns)
# col_names = annual_returns.index
#
# for i, col_name in enumerate(col_names):
#     b = (annual_returns.iloc[[i]]).melt(ignore_index=False).reset_index()  # each row to panel
#     b = b.drop(columns=[b.columns[0]])
#     b = b.rename(columns={'variable': 'year', 'value': col_name})
#     df1 = b[(b['year'] >= 2004) & (b['year'] <= 2020)]
#     x = sm.add_constant(df1.year)
#     y = df1[col_name]
#     lm = sm.OLS(y, x).fit()
#     lm_fit = lm.predict(x)
#     b[col_name] = b[col_name] - lm_fit
#
#     c = pd.DataFrame.copy(b)
#     c = c.add_prefix('lagged_')
#     c['year'] = b['year'] + 1  # returns lagged one year
#     c = c.drop(columns='lagged_year')
#
#     merged_data_returns = pd.merge(merged_data_returns, b, on='year')
#     merged_data_returns = pd.merge(merged_data_returns, c, on='year')
#
# merged_data_returns.insert(0, 'Equity risk premium',
#                            merged_data_returns['S&P 500 (includes dividends)'] - merged_data_returns['3-month T.Bill'])
# merged_data_returns.insert(0, 'Real S&P',
#                            merged_data_returns['S&P 500 (includes dividends)'] - merged_data_returns['Inflation Rate'])
# merged_data_returns.to_csv('data_prepare.csv')
#
# # regression:
# year = pd.Categorical(merged_data_returns.year)
# merged_data_returns = merged_data_returns.set_index(['id', 'year'])
# merged_data_returns['year'] = year
# # state_income = pd.Categorical(merged_data_focus.state_income)
# # merged_data_focus['state_income'] = state_income
# # y_names = ['S&P 500 (includes dividends)', 'Equity risk premium', 'Two-year', 'Three-year']
# y_names = ['lagged_S&P 500 (includes dividends)', 'lagged_Two-year', 'lagged_Three-year']
# # y_names = ['Equity risk premium',  'Real S&P', 'Three-year']
# for i, exog_vars in enumerate(y_names):
#     a = PanelOLS(
#         # dependent=merged_data_focus['parti_rate'],
#         dependent=merged_data_returns['parti_rate_detrend'],
#         exog=sm.add_constant(merged_data_returns[exog_vars]),
#         # entity_effects=True,
#         other_effects=merged_data_returns[['state_zip', 'state_income']]
#     ).fit(
#         cov_type='robust',
#         # cov_type='clustered', cluster_entity=True,
#         # cluster_time=True,
#     )
#     print(a)
#     # table = {
#     #     '(1)': PanelOLS(dependent=a['parti rate' + sheet_names_data[0]],
#     #                     exog=sm.add_constant(a[exog_vars]), entity_effects=True,
#     #                     ).fit(
#     #         # cov_type='robust',
#     #         cov_type='clustered', cluster_entity=True,
#     #         cluster_time=True
#     #     ),
#     #     '(2)': PanelOLS(dependent=a['parti rate' + sheet_names_data[1]],
#     #                     exog=sm.add_constant(a[exog_vars]), entity_effects=True).fit(
#     #         # cov_type='robust',
#     #         cov_type='clustered', cluster_entity=True,
#     #         cluster_time=True
#     #     ),
#     #     '(3)': PanelOLS(dependent=a['parti rate' + sheet_names_data[2]],
#     #                     exog=sm.add_constant(a[exog_vars]), entity_effects=True).fit(
#     #         # cov_type='robust',
#     #         cov_type='clustered', cluster_entity=True,
#     #         cluster_time=True
#     #     ),
#     #     '(4)': PanelOLS(dependent=a['parti rate' + sheet_names_data[3]],
#     #                     exog=sm.add_constant(a[exog_vars]), entity_effects=True).fit(
#     #         # cov_type='robust',
#     #         cov_type='clustered', cluster_entity=True,
#     #         cluster_time=True
#     #     ),
#     #     '(5)': PanelOLS(dependent=a['parti rate' + sheet_names_data[4]],
#     #                     exog=sm.add_constant(a[exog_vars]), entity_effects=True).fit(
#     #         # cov_type='robust',
#     #         cov_type='clustered', cluster_entity=True,
#     #         cluster_time=True
#     #     ),
#     # }
#     # display(results.compare(table))
#     # comparrison = results.compare(table)
#     # summary = comparrison.summary
#     # print(summary.as_latex())

# CRSP data:
data_CRSP = pd.read_csv(r'E:/Users/A2010290/Documents/GitHub/NoShort/data/CRSP.csv')
data_CRSP_sort = data_CRSP.sort_values(by=['GVKEY', 'datadate'])
data_CRSP_sort.datadate = pd.to_datetime(data_CRSP_sort.datadate, format="%Y-%m-%d")
data_CRSP_sort.insert(0, 'year', data_CRSP_sort.datadate.dt.year)
data_CRSP_sort = data_CRSP_sort[data_CRSP_sort.groupby(['GVKEY', 'year'])['trt1m'].transform('count').ge(12)]
data_CRSP_sort = data_CRSP_sort[data_CRSP_sort.fic == 'USA']  # keep only the US companies
data_CRSP_sort.insert(0, 'm_returns', data_CRSP_sort.trt1m / 100 + 1)
annualy = data_CRSP_sort.groupby(['GVKEY', 'year']).prod().reset_index(['GVKEY', 'year'], drop=False)
annualy = annualy.rename(columns={'m_returns': 'y_returns'})
annualy = annualy.drop(annualy.columns[3:], axis=1)
data_CRSP_sort_1 = pd.DataFrame.merge(data_CRSP_sort, annualy)
data_CRSP_sort_1 = data_CRSP_sort_1.fillna(method='ffill')
data_CRSP_sort_1.insert(0, 'month', data_CRSP_sort_1.datadate.dt.month)
data_CRSP_sort_2 = data_CRSP_sort_1[data_CRSP_sort_1.month == 12]
data_CRSP_sort_2 = data_CRSP_sort_2.drop(data_CRSP_sort_2.columns[0], axis=1)
data_CRSP_sort_2.insert(0, 'market_cap', data_CRSP_sort_2.prccm * data_CRSP_sort_2.cshoq)
# removes the state_year with lower than 20 companies
data_CRSP_sort_2 = data_CRSP_sort_2[data_CRSP_sort_2.groupby(['state', 'year'])['market_cap'].transform('count').ge(20)]
# todo: right now is market cap weighted average returns;
#  should construct a state index first before calculating returns
wm = lambda x: np.average(x, weights=data_CRSP_sort_2.loc[x.index, 'market_cap'])
returns_state_year = data_CRSP_sort_2.groupby(
    ['year', 'state']).agg(
    weighted_return_state=('y_returns', wm)
).reset_index(['year', 'state'], drop=False)

returns_state_year = returns_state_year.rename(columns={'state': 'Abbreviation'})
returns_state_year = pd.DataFrame.merge(returns_state_year, match_names)
returns_state_year = returns_state_year.rename(columns={'US State': 'States'})
merged_data = pd.DataFrame.copy(a)
for i in range(4):
    returns_state_year_i = pd.DataFrame.copy(returns_state_year)
    returns_state_year_i['year'] = returns_state_year_i['year'] + i
    returns_state_year_i = returns_state_year_i.rename(columns={'weighted_return_state': 'weighted_return_state' + str(i)})
    merged_data = pd.DataFrame.merge(merged_data, returns_state_year_i, how='left')
merged_data.insert(0, 'two_year', merged_data.weighted_return_state1 * merged_data.weighted_return_state2)
merged_data.insert(0, 'three_year', merged_data.weighted_return_state1 * merged_data.weighted_return_state2 * merged_data.weighted_return_state3)
col_names = annual_returns.index

for i, col_name in enumerate(col_names):
    b = (annual_returns.iloc[[i]]).melt(ignore_index=False).reset_index()  # each row to panel
    b = b.drop(columns=[b.columns[0]])
    b = b.rename(columns={'variable': 'year', 'value': col_name})
    # b['year'] = b['year']  # returns lagged one year
    b['year'] = b['year'] + 1  # returns lagged one year
    merged_data = pd.merge(merged_data, b, on='year')

merged_data['Equity risk premium'] = merged_data['S&P 500 (includes dividends)'] - merged_data['3-month T.Bill']
merged_data['Real S&P'] = merged_data['S&P 500 (includes dividends)'] - merged_data['Inflation Rate']
year = pd.Categorical(merged_data.year)
merged_data = merged_data.set_index(['States', 'year'])
merged_data['year'] = year
y_names = ['weighted_return_state1', 'two_year', 'three_year']
for i, y_var in enumerate(y_names):
    # exog_vars = [y_var, 'Gold*', ' Baa Corporate Bond', 'Real Estate']
    # exog_vars = ['S&P 500 (includes dividends)', y_var]
    exog_vars = [y_var]
    table = {
        '(1)': PanelOLS(dependent=merged_data['parti rate' + sheet_names_data[0]],
                        exog=sm.add_constant(merged_data[exog_vars]), entity_effects=True,
                        time_effects=True,
                        ).fit(
            # cov_type='robust',
            cov_type='clustered', cluster_entity=True,
            cluster_time=True
        ),
        '(2)': PanelOLS(dependent=merged_data['parti rate' + sheet_names_data[1]],
                        exog=sm.add_constant(merged_data[exog_vars]), entity_effects=True,
                        time_effects=True,
                        ).fit(
            # cov_type='robust',
            cov_type='clustered', cluster_entity=True,
            cluster_time=True
        ),
        '(3)': PanelOLS(dependent=merged_data['parti rate' + sheet_names_data[2]],
                        exog=sm.add_constant(merged_data[exog_vars]), entity_effects=True,
                        time_effects=True,
                        ).fit(
            # cov_type='robust',
            cov_type='clustered', cluster_entity=True,
            cluster_time=True
        ),
        # '(4)': PanelOLS(dependent=merged_data['parti rate' + sheet_names_data[3]],
        #                 exog=sm.add_constant(merged_data[exog_vars]), entity_effects=True,
        #                 time_effects=True,
        #                 ).fit(
        #     # cov_type='robust',
        #     cov_type='clustered', cluster_entity=True,
        #     cluster_time=True
        # ),
        # '(5)': PanelOLS(dependent=merged_data['parti rate' + sheet_names_data[4]],
        #                 exog=sm.add_constant(merged_data[exog_vars]), entity_effects=True,
        #                 time_effects=True,
        #                 ).fit(
        #     # cov_type='robust',
        #     cov_type='clustered', cluster_entity=True,
        #     cluster_time=True
        # ),
    }
    display(results.compare(table))
    comparrison = results.compare(table)
    summary = comparrison.summary
    # print(summary.as_latex())


### Norwegian data
import pandas as pd
import pyreadstat as pyreadstat

# merge demographic data (age end of the year) into stock ownership data, and store the number of participants of age groups
ages = np.array([18, 30, 45, 60, 75])
path = r'G:\BI_Research\0209_LaborMarketOutcomes\Data\STATA\STATA2'
a = pyreadstat.read_dta(r"G:\BI_Research\0209_LaborMarketOutcomes\Data\STATA\STATA2\W19_1210_AMELD_STATDATA_2015_M12.dta")
# # Zip code level data:
# path = r'E:\Users\A2010290\Documents\GitHub\NoShort\data\Zip code data'
# data_keep = ['STATE', 'ZIPCODE', 'AGI_CLASS', 'N1', 'N00600']
# merged_data_numbers = []
# merged_data_parti_rate = []
# for filename in os.listdir(path):
#     if filename.endswith('.csv'):
#         print(filename)
#         data_read = pd.read_csv(os.path.join(path, filename))
#         year = int('20' + filename[:2])
#         data_focus = data_read[data_keep]
#         data_focus['STATE'] = data_focus['STATE'].str.upper()
#         zipcode_int = data_focus[data_keep[1]].astype('Int64')
#         data_focus.insert(0, 'year', year)
#         data_focus.insert(0, 'id', data_focus['STATE'] + '_' + zipcode_int.astype(str).str[:5] + '_' + data_focus[
#             data_keep[2]].astype(str))
#         data_focus.insert(0, 'parti_rate', data_focus['N00600'].div(data_focus['N1']))
#         data_focus = data_focus.loc[
#             (zipcode_int != 0) & (data_focus[data_keep[2]].astype(str) != '0') & (data_focus['N1'] != 0)]
#         data_focus_numbers = data_focus[['id', 'year', 'N00600', 'N1']]
#         data_focus_parti_rate = data_focus[['id', 'year', 'parti_rate']]
#         if year == 2005:
#             merged_data_numbers = data_focus_numbers
#             merged_data_parti_rate = data_focus_parti_rate
#         else:
#             merged_data_numbers = merged_data_numbers.append(data_focus_numbers, ignore_index=True)
#             merged_data_parti_rate = merged_data_parti_rate.append(data_focus_parti_rate, ignore_index=True)
#