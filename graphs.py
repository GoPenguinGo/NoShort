### for an individual path:
import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0, T_cohort, dt)

y1 = Z
y2 = parti
y3 = age
y4 = n_parti

fig, ax1 = plt.subplots()
color1 = 'r'
ax1.set_xlabel('time in simulation, one random path')
ax1.set_ylabel('Zt', color = color1)
ax1.plot(t, y1, color = color1, linewidth = 0.8)
ax1.tick_params(axis='y', labelcolor = color1)
ax2 = ax1.twinx()
color2 = 'b'
ax2.set_ylabel('Population holding stocks', color = color2)
ax2.set_ylim([0,1])
ax2.plot(t, y2, color = color2, linewidth = 0.8)
ax2.tick_params(axis='y', labelcolor = color2)
fig.suptitle('Zt and Participation Rate')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and Participation Rate' + '_keep' + '.jpg')
plt.show()

fig, ax1 = plt.subplots()
color1 = 'r'
ax1.set_xlabel('time in simulation, one random path')
ax1.set_ylabel('Zt', color = color1)
ax1.plot(t, y1, color = color1, linewidth = 0.8)
ax1.tick_params(axis='y', labelcolor = color1)
ax2 = ax1.twinx()
color2 = 'b'
ax2.set_ylabel('Average age holding stocks', color = color2)
ax2.set_ylim([0,100])
ax2.plot(t, y3, color = color2, linewidth = 0.8)
ax2.tick_params(axis='y', labelcolor = color2)
fig.suptitle('Zt and Average Participant Age')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and Average Participant Age' + '_keep' + '.jpg')
plt.show()

fig, ax1 = plt.subplots()
color1 = 'r'
ax1.set_xlabel('time in simulation, one random path')
ax1.set_ylabel('Zt', color = color1)
ax1.plot(t, y1, color = color1, linewidth = 0.8)
ax1.tick_params(axis='y', labelcolor = color1)
ax2 = ax1.twinx()
color2 = 'b'
ax2.set_ylabel('% of cohorts holding stocks', color = color2)
ax2.set_ylim([0,1])
ax2.plot(t, y4, color = color2, linewidth = 0.8)
ax2.tick_params(axis='y', labelcolor = color2)
fig.suptitle('Zt and % of Cohorts Participate')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and Cohorts Participate' + '_keep' + '.jpg')
plt.show()


##############################
y1 = Z
y2 = parti_drop
y3 = age_drop
y4 = n_parti_drop

fig, ax1 = plt.subplots()
color1 = 'r'
ax1.set_xlabel('time in simulation, one random path')
ax1.set_ylabel('Zt', color = color1)
ax1.plot(t, y1, color = color1, linewidth = 0.8)
ax1.tick_params(axis='y', labelcolor = color1)
ax2 = ax1.twinx()
color2 = 'b'
ax2.set_ylabel('Population holding stocks, drop', color = color2)
ax2.set_ylim([0,1])
ax2.plot(t, y2, color = color2, linewidth = 0.8)
ax2.tick_params(axis='y', labelcolor = color2)
fig.suptitle('Zt and Participation Rate, if Drop')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and Participation Rate' + '_drop' + '.jpg')
plt.show()

fig, ax1 = plt.subplots()
color1 = 'r'
ax1.set_xlabel('time in simulation, one random path')
ax1.set_ylabel('Zt', color = color1)
ax1.plot(t, y1, color = color1, linewidth = 0.8)
ax1.tick_params(axis='y', labelcolor = color1)
ax2 = ax1.twinx()
color2 = 'b'
ax2.set_ylabel('Average age holding stocks', color = color2)
ax2.set_ylim([0,100])
ax2.plot(t, y3, color = color2, linewidth = 0.8)
ax2.tick_params(axis='y', labelcolor = color2)
fig.suptitle('Zt and Average Participant Age, if Drop')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and Average Participant Age' + '_drop' + '.jpg')
plt.show()

fig, ax1 = plt.subplots()
color1 = 'r'
ax1.set_xlabel('time in simulation, one random path')
ax1.set_ylabel('Zt', color = color1)
ax1.plot(t, y1, color = color1, linewidth = 0.8)
ax1.tick_params(axis='y', labelcolor = color1)
ax2 = ax1.twinx()
color2 = 'b'
ax2.set_ylabel('% of cohorts holding stocks', color = color2)
ax2.set_ylim([0,1])
ax2.plot(t, y4, color = color2, linewidth = 0.8)
ax2.tick_params(axis='y', labelcolor = color2)
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
ax1.set_ylabel('Zt', color = color1)
ax1.plot(t, y1, color = color1, linewidth = 0.8)
ax1.tick_params(axis='y', labelcolor = color1)
ax2 = ax1.twinx()
color2 = 'b'
ax2.set_ylabel('Population longing stocks', color = color2)
ax2.set_ylim([0,1])
ax2.plot(t, y2, color = color2, linewidth = 0.8)
ax2.tick_params(axis='y', labelcolor = color2)
fig.suptitle('Zt and Participation Rate, complete market')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and Participation Rate' + '_comp' + '.jpg')
plt.show()

fig, ax1 = plt.subplots()
color1 = 'r'
ax1.set_xlabel('time in simulation, one random path')
ax1.set_ylabel('Zt', color = color1)
ax1.plot(t, y1, color = color1, linewidth = 0.8)
ax1.tick_params(axis='y', labelcolor = color1)
ax2 = ax1.twinx()
color2 = 'b'
ax2.set_ylabel('Average age longing stocks', color = color2)
ax2.set_ylim([0,100])
ax2.plot(t, y3, color = color2, linewidth = 0.8)
ax2.tick_params(axis='y', labelcolor = color2)
fig.suptitle('Zt and Average Participant Age, complete market')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and Average Participant Age' + '_comp' + '.jpg')
plt.show()

fig, ax1 = plt.subplots()
color1 = 'r'
ax1.set_xlabel('time in simulation, one random path')
ax1.set_ylabel('Zt', color = color1)
ax1.plot(t, y1, color = color1, linewidth = 0.8)
ax1.tick_params(axis='y', labelcolor = color1)
ax2 = ax1.twinx()
color2 = 'b'
ax2.set_ylabel('% of cohorts longing stocks', color = color2)
ax2.set_ylim([0,1])
ax2.plot(t, y4, color = color2, linewidth = 0.8)
ax2.tick_params(axis='y', labelcolor = color2)
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
ax1.set_ylabel('Zt', color = color1)
ax1.plot(t, y1, color = color1, linewidth = 0.8)
ax1.tick_params(axis='y', labelcolor = color1)
ax2 = ax1.twinx()
color2 = 'b'
color3 = 'g'
ax2.set_ylabel('Interest Rate', color = color2)
ax2.set_ylim([0,0.05])
ax2.plot(t, y2, color = color2, linewidth = 0.5)
ax2.plot(t, y3, color = color3, linewidth = 0.5)
ax2.tick_params(axis='y', labelcolor = color2)
fig.suptitle('Zt and Interest Rate')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and interest rate' + '.jpg')
plt.show()



y2 = theta_drop * sigma_S
y3 = theta_comp * sigma_S

fig, ax1 = plt.subplots()
color1 = 'r'
ax1.set_xlabel('time in simulation, one random path')
ax1.set_ylabel('Zt', color = color1)
ax1.plot(t, y1, color = color1, linewidth = 0.8)
ax1.tick_params(axis='y', labelcolor = color1)
ax2 = ax1.twinx()
color2 = 'b'
color3 = 'g'
ax2.set_ylabel('equity risk premium', color = color2)
ax2.set_ylim([-0.01,0.02])
ax2.plot(t, y2, color = color2, linewidth = 0.5)
ax2.plot(t, y3, color = color3, linewidth = 0.5)
ax2.tick_params(axis='y', labelcolor = color2)
fig.suptitle('Zt and Equity Risk Premium')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and equity risk premium' + '.jpg')
plt.show()


f_dt_drop = f_drop * dt
weighted_Delta_drop = Delta_drop * f_drop * dt
invest_drop = pi_drop > 0
parti_rate_drop = invest_drop * cohort_size

varlist = [f_dt_drop, weighted_Delta_drop, parti_rate_drop]
var = parti_rate_drop
# y_label = 'wealth share of age groups'
# y_label = 'bias of age groups'
# y_label = 'participation rate of age groups'
y_label = 'participation composition of age groups'
y1 = Z
y2 = np.sum(var[:, tau_cutoff1:], axis = 1)
y3 = np.sum(var[:, tau_cutoff2:tau_cutoff1], axis = 1)
y4 = np.sum(var[:, tau_cutoff3:tau_cutoff2], axis = 1)
y5 = np.sum(var[:, :tau_cutoff3], axis = 1)

fig, ax1 = plt.subplots()
color1 = 'r'
ax1.set_xlabel('time in simulation, one random path')
ax1.set_ylabel('Zt', color = color1)
ax1.plot(t, y1, color = color1, linewidth = 0.8)
ax1.tick_params(axis='y', labelcolor = color1)
ax2 = ax1.twinx()
color2 = 'b'
color3 = 'g'
color4 = 'y'
color5 = 'm'
ax2.set_ylabel(y_label, color = color2)
ax2.set_ylim([0,1])
ax2.plot(t, y2, color = color2, linewidth = 0.5)
ax2.plot(t, y3, color = color3, linewidth = 0.5)
ax2.plot(t, y4, color = color4, linewidth = 0.5)
ax2.plot(t, y5, color = color5, linewidth = 0.5)
ax2.tick_params(axis='y', labelcolor = color2)
fig.suptitle('Zt and ' + y_label + '(blue, green, yellow, magenta)')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and ' + y_label + '(absolute).jpg')
plt.show()