### for an individual path:
import matplotlib.pyplot as plt
import numpy as np

# define the colors:
color1 = 'black'
color2 = 'mediumblue'
color3 = 'darkgreen'
color4 = 'orange'
color5 = 'red'
color6 = 'b'
color7 = 'g'

#######################################
########## ONE RANDOM PATH ############
#######################################
# the x-axis
t = np.arange(0, T_cohort, dt)
y0 = Z

#######################################
############# GRAPH ONE ###############
#######################################
# illustrate the learning process
nn = 3
length = len(t)
Delta_time_series = np.zeros((nn, length))
for i in range(3):
    start = int((i + 1) * 100 * (1 / dt))
    for j in range(length):
        if j < start:
            Delta_time_series[i, j] = np.nan
        else:
            cohort_rank = length - (j - start) - 1
            Delta_time_series[i, j] = Delta[j, cohort_rank]

y11 = Delta_time_series[0]
y12 = Delta_time_series[1]
y13 = Delta_time_series[2]

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.set_xlabel('Time in simulation, one random path')
ax1.set_ylabel('Zt', color=color5)
ax1.plot(t, y0, color=color5, linewidth=0.5)
ax1.tick_params(axis='y', labelcolor=color5)
ax2 = ax1.twinx()
ax2.set_ylabel('Bias in belief and learning', color=color2)
ax2.set_ylim([-0.5, 0.5])
ax2.plot(t, y11, color=color2, linewidth=0.4)
ax2.plot(t, y12, color=color3, linewidth=0.4)
ax2.plot(t, y13, color=color4, linewidth=0.4)
ax2.tick_params(axis='y', labelcolor=color2)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and bias time series' + '.png', dpi=500)
plt.show()

#######################################
############# GRAPH TWO ###############
#######################################
# plot the market price of risk
y21 = theta_comp
y22 = theta_drop

fig, ax1 = plt.subplots(figsize=(15, 5))
ax1.set_xlabel('Time in simulation, one random path')
ax1.set_ylabel('Zt', color=color5)
ax1.plot(t, y0, color=color5, linewidth=0.5)
ax1.tick_params(axis='y', labelcolor=color5)
ax2 = ax1.twinx()
ax2.set_ylabel('Market price of risk', color=color2)
ax2.set_ylim([-1, 1])
ax2.plot(t, y21, color=color2, linewidth=0.4, label='Complete market')
ax2.plot(t, y22, color=color3, linewidth=0.4, label='Short-sale constraint')
ax2.hlines(sigma_Y, xmin=0, xmax=500, color='purple', linestyles='--', linewidth=0.8, label='Representative agent')
ax2.tick_params(axis='y', labelcolor=color2)
plt.legend()
# fig.suptitle('Zt and Market Price of Risk')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and market price of risk' + '.png', dpi=500)
plt.show()

# plot the interest rate
y31 = r_comp
y32 = r_drop
y33 = rho + mu_Y - sigma_Y ** 2

fig, ax1 = plt.subplots(figsize=(15, 5))
ax1.set_xlabel('Time in simulation, one random path')
ax1.set_ylabel('Zt', color=color5)
ax1.plot(t, y0, color=color5, linewidth=0.5)
ax1.tick_params(axis='y', labelcolor=color5)
ax2 = ax1.twinx()
ax2.set_ylabel('Interest rate (annual)', color=color2)
ax2.set_ylim([0, 0.05])
ax2.plot(t, y31, color=color2, linewidth=0.4, label='Complete market')
ax2.plot(t, y32, color=color3, linewidth=0.4, label='Short-sale constraint')
ax2.hlines(y33, xmin=0, xmax=500, color='purple', linestyles='--', linewidth=0.8, label='Representative agent')
ax2.tick_params(axis='y', labelcolor=color2)
plt.legend()
# fig.suptitle('Zt and Market Price of Risk')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and interest rate' + '.png', dpi=500)
plt.show()

#######################################
############ GRAPH THREE ##############
#######################################
y41 = Delta_bar_parti_comp
y42 = Delta_bar_parti_drop / f_parti_drop
y43 = f_parti_drop
y44 = parti_drop
fig, ax1 = plt.subplots(figsize=(15, 5))
ax1.set_xlabel('Time in simulation, one random path')
ax1.set_ylabel('Zt', color=color5)
ax1.plot(t, y0, color=color5, linewidth=0.5)
ax1.tick_params(axis='y', labelcolor=color5)
ax2 = ax1.twinx()
ax2.set_ylabel('Consumption weighted bias', color=color2)
ax2.set_ylim([-0.5, 1])
ax2.plot(t, y41, color='purple', linewidth=0.4, linestyle ='--', label='Market view, complete market')
ax2.plot(t, y42, color=color3, linewidth=0.4, label='Market view, no shorting')
ax2.plot(t, y43, color=color4, linewidth=0.4, label='Participant consumption share')
ax2.plot(t, y44, color=color2, linewidth=0.4, label='Participation rate')
ax2.tick_params(axis='y', labelcolor=color2)
plt.legend()
# fig.suptitle('Zt and market view')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and market bias' + '.png', dpi=500)
plt.show()

#######################################
############ GRAPH  FOUR ##############
#######################################
# who are the investors
invest_drop = pi_drop > 0
parti_rate_drop = invest_drop * cohort_size
var = parti_rate_drop
y_label = 'participation composition of age groups'
y51 = np.sum(var[:, tau_cutoff1:], axis=1)
y52 = np.sum(var[:, tau_cutoff2:], axis=1)
y53 = np.sum(var[:, tau_cutoff3:], axis=1)
y54 = np.sum(var[:, ], axis=1)
fig, ax1 = plt.subplots(figsize=(15, 5))
ax1.set_ylabel('Participation composition of age groups', color=color2)
ax1.set_ylim([0, 1])
ax1.fill_between(t, y51, color = 'steelblue', linewidth = 0.4, label = '20 < Age <= 35, youngest quartile')
ax1.fill_between(t, y52, y51, color = 'darkseagreen', linewidth = 0.4, label = '35 < Age <= 55')
ax1.fill_between(t, y53, y52, color = 'moccasin', linewidth = 0.4, label= '55 < Age <= 89')
ax1.fill_between(t, y54, y53, color = 'pink', linewidth = 0.4, label= 'Age > 89, oldest quartile')
# ax1.fill_between(t, y51, color='lavender', linewidth=0.4, label='20 < Age <= 35, youngest quartile')
# ax1.fill_between(t, y52, y51, color='lightsteelblue', linewidth=0.4, label='35 < Age <= 55')
# ax1.fill_between(t, y53, y52, color='steelblue', linewidth=0.4, label='55 < Age <= 89')
# ax1.fill_between(t, y54, y53, color='royalblue', linewidth=0.4, label='Age > 89, oldest quartile')
ax1.tick_params(axis='y', labelcolor=color2)
plt.legend()
ax2 = ax1.twinx()
ax2.set_xlabel('Time in simulation, one random path')
ax2.set_ylabel('Zt', color=color5)
ax2.plot(t, y0, color=color5, linewidth=0.5)
ax2.tick_params(axis='y', labelcolor=color5)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and participation composition' + '.png', dpi = 500)
plt.show()



# who want to short
short_popu_drop = short_drop * cohort_size
var = short_popu_drop
y51 = np.sum(var[:, tau_cutoff1:], axis=1)
y52 = np.sum(var[:, tau_cutoff2:], axis=1)
y53 = np.sum(var[:, tau_cutoff3:], axis=1)
y54 = np.sum(var[:, ], axis=1)
fig, ax1 = plt.subplots(figsize=(15, 5))
ax1.set_ylabel('Short-sale demand composition of age groups', color=color2)
ax1.set_ylim([0, 0.1])
ax1.fill_between(t, y51, color = 'steelblue', linewidth = 0.4, label = '20 < Age <= 35, youngest quartile')
ax1.fill_between(t, y52, y51, color = 'darkseagreen', linewidth = 0.4, label = '35 < Age <= 55')
ax1.fill_between(t, y53, y52, color = 'moccasin', linewidth = 0.4, label= '55 < Age <= 89')
ax1.fill_between(t, y54, y53, color = 'pink', linewidth = 0.4, label= 'Age > 89, oldest quartile')
# ax1.fill_between(t, y51, color='lavender', linewidth=0.4, label='20 < Age <= 35, youngest quartile')
# ax1.fill_between(t, y52, y51, color='lightsteelblue', linewidth=0.4, label='35 < Age <= 55')
# ax1.fill_between(t, y53, y52, color='steelblue', linewidth=0.4, label='55 < Age <= 89')
# ax1.fill_between(t, y54, y53, color='royalblue', linewidth=0.4, label='Age > 89, oldest quartile')
ax1.tick_params(axis='y', labelcolor=color2)
plt.legend()
ax2 = ax1.twinx()
ax2.set_xlabel('Time in simulation, one random path')
ax2.set_ylabel('Zt', color=color5)
ax2.plot(t, y0, color=color5, linewidth=0.5)
ax2.tick_params(axis='y', labelcolor=color5)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and short sale composition' + '.png', dpi = 500)
plt.show()

















y2 = popu_long_collect
y22 = popu_long_free
y23 = theta_comp
y3 = age
y4 = n_parti
tail = '_rich_free'

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.set_xlabel('Time in simulation, one random path')
ax1.set_ylabel('Zt', color=color5)
ax1.plot(t, y1, color=color5, linewidth=0.5)
ax1.tick_params(axis='y', labelcolor=color5)
ax2 = ax1.twinx()
# ax2.set_ylabel('Population investing in stocks', color = color2)
ax2.set_ylabel('Bias in belief and learning', color=color2)
ax2.set_ylim([-0.5, 0.5])
ax2.plot(t, y22, color=color2, linewidth=0.4)
ax2.plot(t, y23, color=color3, linewidth=0.4)
ax2.plot(t, y24, color=color4, linewidth=0.4)
ax2.tick_params(axis='y', labelcolor=color2)
# fig.suptitle('Zt and Shorting Rate')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig('Zt and Shorting Rate Comparison' + tail + '.jpg')
plt.savefig('Zt and bias time series' + '.png', dpi=500)
plt.show()

fig, ax1 = plt.subplots()
color1 = 'r'
ax1.set_xlabel('time in simulation, one random path')
ax1.set_ylabel('Zt', color=color1)
ax1.plot(t, y1, color=color1, linewidth=0.8)
ax1.tick_params(axis='y', labelcolor=color1)
ax2 = ax1.twinx()
color2 = 'b'
ax2.set_ylabel('Average age holding stocks', color=color2)
ax2.set_ylim([0, 100])
ax2.plot(t, y3, color=color2, linewidth=0.8)
ax2.tick_params(axis='y', labelcolor=color2)
fig.suptitle('Zt and Average Participant Age')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and Average Participant Age' + tail + '.jpg')
plt.show()

fig, ax1 = plt.subplots()
color1 = 'r'
ax1.set_xlabel('time in simulation, one random path')
ax1.set_ylabel('Zt', color=color1)
ax1.plot(t, y1, color=color1, linewidth=0.8)
ax1.tick_params(axis='y', labelcolor=color1)
ax2 = ax1.twinx()
color2 = 'b'
ax2.set_ylabel('% of cohorts holding stocks', color=color2)
ax2.set_ylim([0, 1])
ax2.plot(t, y4, color=color2, linewidth=0.8)
ax2.tick_params(axis='y', labelcolor=color2)
fig.suptitle('Zt and % of Cohorts Participate')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and Cohorts Participate' + tail + '.jpg')
plt.show()

##############################
y1 = Z
y2 = parti_drop
y3 = age_drop
y4 = n_parti_drop

fig, ax1 = plt.subplots()
color1 = 'r'
ax1.set_xlabel('time in simulation, one random path')
ax1.set_ylabel('Zt', color=color1)
ax1.plot(t, y1, color=color1, linewidth=0.8)
ax1.tick_params(axis='y', labelcolor=color1)
ax2 = ax1.twinx()
color2 = 'b'
ax2.set_ylabel('Population holding stocks, drop', color=color2)
ax2.set_ylim([0, 1])
ax2.plot(t, y2, color=color2, linewidth=0.8)
ax2.tick_params(axis='y', labelcolor=color2)
fig.suptitle('Zt and Participation Rate, if Drop')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and Participation Rate' + '_drop' + '.jpg')
plt.show()

fig, ax1 = plt.subplots()
color1 = 'r'
ax1.set_xlabel('time in simulation, one random path')
ax1.set_ylabel('Zt', color=color1)
ax1.plot(t, y1, color=color1, linewidth=0.8)
ax1.tick_params(axis='y', labelcolor=color1)
ax2 = ax1.twinx()
color2 = 'b'
ax2.set_ylabel('Average age holding stocks', color=color2)
ax2.set_ylim([0, 100])
ax2.plot(t, y3, color=color2, linewidth=0.8)
ax2.tick_params(axis='y', labelcolor=color2)
fig.suptitle('Zt and Average Participant Age, if Drop')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and Average Participant Age' + '_drop' + '.jpg')
plt.show()

fig, ax1 = plt.subplots()
color1 = 'r'
ax1.set_xlabel('time in simulation, one random path')
ax1.set_ylabel('Zt', color=color1)
ax1.plot(t, y1, color=color1, linewidth=0.8)
ax1.tick_params(axis='y', labelcolor=color1)
ax2 = ax1.twinx()
color2 = 'b'
ax2.set_ylabel('% of cohorts holding stocks', color=color2)
ax2.set_ylim([0, 1])
ax2.plot(t, y4, color=color2, linewidth=0.8)
ax2.tick_params(axis='y', labelcolor=color2)
fig.suptitle('Zt and % of Cohorts Participate, if Drop')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and Cohorts Participate' + '_drop' + '.jpg')
plt.show()

y1 = Z
y2 = parti_comp
y3 = age_comp
y4 = n_parti_comp

fig, ax1 = plt.subplots()
color1 = 'r'
ax1.set_xlabel('time in simulation, one random path')
ax1.set_ylabel('Zt', color=color1)
ax1.plot(t, y1, color=color1, linewidth=0.8)
ax1.tick_params(axis='y', labelcolor=color1)
ax2 = ax1.twinx()
color2 = 'b'
ax2.set_ylabel('Population longing stocks', color=color2)
ax2.set_ylim([0, 1])
ax2.plot(t, y2, color=color2, linewidth=0.8)
ax2.tick_params(axis='y', labelcolor=color2)
fig.suptitle('Zt and Participation Rate, complete market')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and Participation Rate' + '_comp' + '.jpg')
plt.show()

fig, ax1 = plt.subplots()
color1 = 'r'
ax1.set_xlabel('time in simulation, one random path')
ax1.set_ylabel('Zt', color=color1)
ax1.plot(t, y1, color=color1, linewidth=0.8)
ax1.tick_params(axis='y', labelcolor=color1)
ax2 = ax1.twinx()
color2 = 'b'
ax2.set_ylabel('Average age longing stocks', color=color2)
ax2.set_ylim([0, 100])
ax2.plot(t, y3, color=color2, linewidth=0.8)
ax2.tick_params(axis='y', labelcolor=color2)
fig.suptitle('Zt and Average Participant Age, complete market')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and Average Participant Age' + '_comp' + '.jpg')
plt.show()

fig, ax1 = plt.subplots()
color1 = 'r'
ax1.set_xlabel('time in simulation, one random path')
ax1.set_ylabel('Zt', color=color1)
ax1.plot(t, y1, color=color1, linewidth=0.8)
ax1.tick_params(axis='y', labelcolor=color1)
ax2 = ax1.twinx()
color2 = 'b'
ax2.set_ylabel('% of cohorts longing stocks', color=color2)
ax2.set_ylim([0, 1])
ax2.plot(t, y4, color=color2, linewidth=0.8)
ax2.tick_params(axis='y', labelcolor=color2)
fig.suptitle('Zt and % of Cohorts Participate, complete market')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and Cohorts Participate' + '_comp' + '.jpg')
plt.show()

#############################
#### comparative graphs

y1 = Z
y2 = r_drop
y3 = r_comp

fig, ax1 = plt.subplots()
color1 = 'r'
ax1.set_xlabel('time in simulation, one random path')
ax1.set_ylabel('Zt', color=color1)
ax1.plot(t, y1, color=color1, linewidth=0.8)
ax1.tick_params(axis='y', labelcolor=color1)
ax2 = ax1.twinx()
color2 = 'b'
color3 = 'g'
ax2.set_ylabel('Interest Rate', color=color2)
ax2.set_ylim([0, 0.05])
ax2.plot(t, y2, color=color2, linewidth=0.5)
ax2.plot(t, y3, color=color3, linewidth=0.5)
ax2.tick_params(axis='y', labelcolor=color2)
fig.suptitle('Zt and Interest Rate')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and interest rate' + '.jpg')
plt.show()

y2 = theta_drop
y22 = theta_free
y23 = theta_comp

fig, ax1 = plt.subplots()
color1 = 'r'
ax1.set_xlabel('time in simulation, one random path')
ax1.set_ylabel('Zt', color=color1)
ax1.plot(t, y1, color=color1, linewidth=0.8)
ax1.tick_params(axis='y', labelcolor=color1)
ax2 = ax1.twinx()
color2 = 'b'
color3 = 'g'
color4 = 'm'
ax2.set_ylabel('equity risk premium', color=color2)
ax2.set_ylim([-0.5, 1])
ax2.plot(t, y2, color=color2, linewidth=0.5)
ax2.plot(t, y22, color=color3, linewidth=0.5)
ax2.plot(t, y23, color=color4, linewidth=0.5)
ax2.tick_params(axis='y', labelcolor=color2)
fig.suptitle('Zt and Equity Risk Premium, drop=b, free=g, comp=m')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and equity risk premium' + '.jpg')
plt.show()

# f_dt_drop = f_drop * dt
# weighted_Delta_drop = Delta_drop * f_drop * dt
# invest_drop = pi_drop > 0
# parti_rate_drop = invest_drop * cohort_size

varlist = [f_dt_drop, weighted_Delta_drop, parti_rate_drop]

f_dt_free = f_free * dt
weighted_Delta_drop = Delta_drop * f_drop * dt
invest_drop = pi_drop > 0
parti_rate_drop = invest_drop * cohort_size

var = f_dt_free
y_label = 'wealth share of age groups'
# y_label = 'bias of age groups'
# y_label = 'participation rate of age groups'
# y_label = 'participation composition of age groups'
y1 = Z
y2 = np.sum(var[:, tau_cutoff1:], axis=1)
y3 = np.sum(var[:, tau_cutoff2:tau_cutoff1], axis=1)
y4 = np.sum(var[:, tau_cutoff3:tau_cutoff2], axis=1)
y5 = np.sum(var[:, :tau_cutoff3], axis=1)

fig, ax1 = plt.subplots()
color1 = 'r'
ax1.set_xlabel('time in simulation, one random path')
ax1.set_ylabel('Zt', color=color1)
ax1.plot(t, y1, color=color1, linewidth=0.8)
ax1.tick_params(axis='y', labelcolor=color1)
ax2 = ax1.twinx()
color2 = 'b'
color3 = 'g'
color4 = 'y'
color5 = 'm'
ax2.set_ylabel(y_label, color=color2)
ax2.set_ylim([0, 1])
ax2.plot(t, y2, color=color2, linewidth=0.5)
ax2.plot(t, y3, color=color3, linewidth=0.5)
ax2.plot(t, y4, color=color4, linewidth=0.5)
ax2.plot(t, y5, color=color5, linewidth=0.5)
ax2.tick_params(axis='y', labelcolor=color2)
fig.suptitle('Zt and ' + y_label + '(blue, green, yellow, magenta)')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and ' + y_label + tail + '.jpg')
plt.show()
