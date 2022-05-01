import numpy as np

class Believe:
    def __init__(self, Vhat) -> None:
        self.Delta_s_t = np.zeros(1)
        self.Vhat = Vhat

class BaseCohorts:

    def __init__(self, Nc: int, nu: float, dt: float, beta: float, Vhat: float):
        self.Nc = Nc
        self.reduction = np.exp(-nu * dt)
        self.IntVec = nu * beta
        self.believe = Believe(Vhat)

    def evolve(self): 
        """Evolve one time unit"""
        self.update_believe()
        self.decide_investment()

    def update_believe(self):
        ...

    def decide_investment(self): 
        ...
