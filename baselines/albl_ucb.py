import numpy as np

class UCBBatchAL:
    def __init__(self, n, c=1.0, pseudo_count=1e-6):
        """
        Batch active learning with UCB over strategies.
        
        Parameters
        ----------
        n : int, number of arms i.e. strategies in the portfolio
        c : float, default=1.0
            UCB exploration parameter.
        pseudo_count : float, default=1e-6
            Tiny count to avoid division by zero in UCB formula.
        """
        self.K = n
        self.c = c
        self.counts = [pseudo_count] * self.K
        self.values = [0.0] * self.K
        self.prev_acc = 0.0

    def select_strategy(self, round_num):
        '''
        Responsible to select the action for the next time step
        '''
        ucb_values = [self.values[i] + self.c * np.sqrt(np.log(max(1, round_num+1)) / self.counts[i])
                      for i in range(self.K)]
        max_val = np.max(ucb_values)
        candidates = [i for i, val in enumerate(ucb_values) if val == max_val]
        return np.random.choice(candidates).item()

    def update_values(self, strategy_idx, acc):
        '''
        Based on the last action and the observed reward, update UCB statistics
        '''
        ## First round so just store the accuracy
        if strategy_idx == -1:
            self.prev_acc = acc
            return
        # Difference in accuracy is the reward for the UCB
        reward = acc - self.prev_acc

        #Update the UCB statistics for the chosen arm in the round
        self.counts[strategy_idx] += 1
        n = self.counts[strategy_idx]
        self.values[strategy_idx] += (reward - self.values[strategy_idx]) / n
        self.prev_acc = acc