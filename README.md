# No Short in Sight
Paul Ehling, Christian Heyerdahl-Larsen, and Zeshu XU

# Install dependencies
```shell
pip install -r requirements.txt
```

# Execute the cohort simulation
```
python3 main.py
```

# Linter
To have a consistent coding style, use `black` to lint the code before commit

```shell
black main.py
```

# Function descriptions
1. **param.py** defines the parameters used in the simulations, and prepares matrix to store the results:
   1. additionally, **param_mix.py** defines the parameters used for the _mix_ scenario;
2. **stats.py** defines some base functions used repeatedly throughout the simulations, including:
   1. post_var(): calculate the posterior variance, correspond to eq(6);
   2. shocks(): calculate Zt and Yt from shocks dZt;
   3. tau_calculator(): calculate tau for the cohorts in the economy;
3. **solver.py** defines some base solver functions used repeatedly throughout the simulations, including:
   1. bisection(): Bisection method to solve for any the solution in any equation;
   2. solve_theta(): combines with bisection(), contains RHS - LHS of the eq(22), used to iteratively solve market-clearing price of risk, ie. theta;
   3. find_the_rich(): finds the agents that make the top x% the richest population in the economy, and they can short;
   4. bisection_partial_constraint() and solve_theta_partial_constraint() are similar to 3.i and 3.ii, but takes more arguments;
4. **cohort_builder.py** defines functions that build up an OLG economy:
   1. build_cohorts_SI(): mainly for the _reentry_ scenario;
   2. build_cohorts_mix_type(): mainly for the _mix_ scenario;
5. **cohort_simulator.py** defines functions that simulate the OLG economy forward:
   1. simulate_cohorts_SI(): mainly for the _reentry_ scenario;
   2. simulate_cohorts_mean_vola(): mainly for the _reentry_ scenario, saves only time-series mean and variance;
   3 simulate_cohorts_mix_type(): mainly for the _mix_ scenario;
   4. simulate_mean_vola_mix_type(): mainly for the _mix_ scenario, saves only time-series mean and variance;
6. **simulation.py** defines functions that bundle 4 and 5 in one function (SI, mean_vola, SI_mix, mean_vola_mix);
7. **main_multiprocessing.py** runs a large number of simulation and store data for figures 4, 8-10;
8. **tab3_multiprocessing.py** runs a large number of rounds of simulation and store data for table3;
9. **main.py** is the main file that:
   1. generates data for figures based on single paths;
   2. generate graphs based on data from **main_multiprocessing.py**
   3. generate tables based on data from **tab3_multiprocessing.py**
