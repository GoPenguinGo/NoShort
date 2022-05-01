import numpy as np

class Macro:
    def __init__(self, dt: float, T_cohort: float, mu_Y: float, sigma_Y: float) -> None:
        """Macro-economic

        Args:
            dt (float): time incremental
            T_cohort (float): time horizon to keep track of cohorts
            mu_Y (float): mean of aggregate output growth
            sigma_Y (float): sd of aggregate output growth
        """
        self.dt = dt
        self.T_cohort = T_cohort
        self.Nt = int(T_cohort / dt)
        self.dZt =  self.dt**0.5 * np.random.randn(self.Nt - 1)
        self.Zt = np.insert(np.cumsum(self.dZt), 0, 0)
        self.mu_Y = mu_Y
        self.sigma_Y = sigma_Y
        self.yg = (mu_Y - 0.5 * sigma_Y**2) * dt + sigma_Y * self.dZt
        self.Yt = np.insert(np.exp(np.cumsum(self.yg)), 0, 1)

    



# class BaseCohorts:
#     a = "C"

#     def __init__(self, t: float, T: float) -> None:
#         self.t  = t
#         self.T = T
#         self.believe = ()
#         self.ratio = ()

#     @staticmethod
#     def print():
#         print("HH")

#     @classmethod
#     def print(cls):
#         cls.a


#     def one_year(self):
#         10 ** self.update_believe()
#         self.invest()
#         self.t += 1

#         return self.t

#     def update_believe(self) -> int:
#         self.believe
#         return self.believe

#     def invest(self):
#         print("Base")
#         self.ratio

# class NewBelieveCohort(BaseCohorts):
#     def update_believe(self) -> int:
#         print("Do something different.")
#         return self.believe + 0.5

# class NewerBelieveCohort(NewBelieveCohort):
#     ...

# class DropCohort(BaseCohorts):

#     def __init__(self, t: float, T: float, b: float) -> None:
#         super().__init__(t, T)
#         self.b = b

#     def invest(self):
#         print("Drop")

#         self.ratio

# if __name__ == "__main__":

#     macro = Macro(z_t = 1, sigma_y=1)
#     print(macro.y_t)

#     cohort = DropCohort(0, 10, 1)

#     for i in range(100):
#         t = cohort.one_year()