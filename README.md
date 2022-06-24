# Learning from Experience no Shorting
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
1. **param.py** defines the parameters used in the simulations, and prepares matrix to store the results;
2. **stats.py** defines some small functions used throughout the simulations, including:
   1. post_var(): calculate the posterior variance, correspond to eq(2);
   2. shocks(): calculate Zt and Yt from shocks dZt;
   3. tau_calculator(): calculate tau for the cohorts in the economy;
   4. good_times(): information indicator;
3. **solver.py** defines some small solver functions, including:
   1. bisection(): Bisection method to solve for any the solution in any equation;
   2. solve_theta(): combines with bisection(), contains RHS - LHS of the eq(24), used to iteratively solve market-clearing price of risk, ie. theta;
   3. find_the_rich(): finds the agents that make the top 5% richest population in the economy, and they can short;
   4. bisection_partial_constraint() and solve_theta_partial_constraint() are similar to 3.i and 3.ii, but takes more arguments;
4. **cohort_builder.py** defines a function that builds up an OLG economy;
5. **cohort_simulator.py** defines a function that simulates the OLG economy forward;
6. **simulation.py** defines a function that wraps 4 and 5 in one function;
7. **V_hat_experiment.py** changes V_hat, or initial variance, and generates graphs to illustrate the simulation results;
8. **main.py** is the main loop that simulates the economy, and generates graphs to illustrate the simulation results.
