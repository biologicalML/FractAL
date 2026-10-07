"""
ALBL algorithm with Exp4.P bandit solver
"""

import numpy as np

class ALBLExp4P:
    """
    Parameters
    ----------
    n : int
        Number of experts, i.e. strategies in the portfolio.
    al_batch_size: int
        To assign the value to p_min = 1/B
    """

    def __init__(self, n: int, al_batch_size: int):
        self.n = n
        self.w = np.ones(n) # Equal initial weight to all experts
        self.pmin = 1.0/ al_batch_size
        self.p = None
        self.last_action = None

    def suggest_action(self) -> np.ndarray:
        """Return the next action to play, one of the portfolio strategies.

        Raises
        ------
        RuntimeError
            If called twice without an intervening update().
        """
        assert self.p is None, "suggest_action() called twice without update() in between"
        self.p = (1 - self.pmin * self.n) * self.w / np.sum(self.w) + self.pmin
        action = np.random.choice(self.n, p=self.p) #Randomly sample from the experts, based on their individual weights/probabilities
        self.last_action = action
        return action

    def update(self, score: float, round_num: int) -> None:
        """Feed the scalar score for the last action played.
        
        Raises
        ------
        RuntimeError
            If called before suggest_action().
        """
        # Ignore for the first round as the method didn't take any action
        if round_num == 0:
            return
        assert self.p is not None, "update() called before suggest_action()"
        reward = score/ self.p[self.last_action]
        y_hat = np.zeros(self.n)
        y_hat[self.last_action] = reward
        v_hat = 1/self.p
        w_new = self.w * np.exp(0.5 * self.pmin *(y_hat + v_hat * self.pmin))
        self.w  = w_new
        self.p = None
        self.last_action = None