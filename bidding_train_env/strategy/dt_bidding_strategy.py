import time
import gin
import numpy as np
import os
import psutil
# from saved_model.DTtest.dt import DecisionTransformer
from bidding_train_env.baseline.dt.dt import DecisionTransformer
from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy
import torch
import pickle
import math
import random


class DtBiddingStrategy(BaseBiddingStrategy):
    """
    Decision-Transformer-PlayerStrategy
    """

    def __init__(self, budget=100, name="Decision-Transformer-PlayerStrategy", cpa=2, category=1):
        super().__init__(budget, name, cpa, category)
        file_name = os.path.dirname(os.path.realpath(__file__))
        dir_name = os.path.dirname(file_name)
        dir_name = os.path.dirname(dir_name)
        model_path = os.path.join(dir_name, "saved_model", "DTtest", "dt.pt")
        picklePath = os.path.join(dir_name, "saved_model", "DTtest", "normalize_dict.pkl")
        self.scale = 200
        self.target_return = budget / cpa / self.scale * 4
        with open(picklePath, 'rb') as f:
            normalize_dict = pickle.load(f)
        self.model = DecisionTransformer(state_dim=16, act_dim=1, state_mean=normalize_dict["state_mean"],
                                         state_std=normalize_dict["state_std"])
        self.model.load_net(model_path)

    def reset(self):
        self.remaining_budget = self.budget
        self.target_return = self.budget / self.cpa / self.scale * 4

    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost):
        """
        Bids for all the opportunities in a delivery period

        parameters:
         @timeStepIndex: the index of the current decision time step.
         @pValues: the conversion action probability.
         @pValueSigmas: the prediction probability uncertainty.
         @historyPValueInfo: the history predicted value and uncertainty for each opportunity.
         @historyBid: the advertiser's history bids for each opportunity.
         @historyAuctionResult: the history auction results for each opportunity.
         @historyImpressionResult: the history impression result for each opportunity.
         @historyLeastWinningCosts: the history least wining costs for each opportunity.

        return:
            Return the bids for all the opportunities in the delivery period.
        """
        time_left = (48 - timeStepIndex) / 48
        budget_left = self.remaining_budget / self.budget if self.budget > 0 else 0
        history_xi = [result[:, 0] for result in historyAuctionResult]
        history_pValue = [result[:, 0] for result in historyPValueInfo]
        history_conversion = [result[:, 1] for result in historyImpressionResult]

        historical_xi_mean = np.mean([np.mean(xi) for xi in history_xi]) if history_xi else 0

        historical_conversion_mean = np.mean(
            [np.mean(reward) for reward in history_conversion]) if history_conversion else 0

        historical_LeastWinningCost_mean = np.mean(
            [np.mean(price) for price in historyLeastWinningCost]) if historyLeastWinningCost else 0

        historical_pValues_mean = np.mean([np.mean(value) for value in history_pValue]) if history_pValue else 0

        historical_bid_mean = np.mean([np.mean(bid) for bid in historyBid]) if historyBid else 0

        def mean_of_last_n_elements(history, n):
            last_three_data = history[max(0, n - 3):n]
            if len(last_three_data) == 0:
                return 0
            else:
                return np.mean([np.mean(data) for data in last_three_data])

        last_three_xi_mean = mean_of_last_n_elements(history_xi, 3)
        last_three_conversion_mean = mean_of_last_n_elements(history_conversion, 3)
        last_three_LeastWinningCost_mean = mean_of_last_n_elements(historyLeastWinningCost, 3)
        last_three_pValues_mean = mean_of_last_n_elements(history_pValue, 3)
        last_three_bid_mean = mean_of_last_n_elements(historyBid, 3)

        current_pValues_mean = np.mean(pValues)
        current_pv_num = len(pValues)

        historical_pv_num_total = sum(len(bids) for bids in historyBid) if historyBid else 0
        last_three_ticks = slice(max(0, timeStepIndex - 3), timeStepIndex)
        last_three_pv_num_total = sum(
            [len(historyBid[i]) for i in range(max(0, timeStepIndex - 3), timeStepIndex)]) if historyBid else 0

        test_state = np.array([
            time_left, budget_left, historical_bid_mean, last_three_bid_mean,
            historical_LeastWinningCost_mean, historical_pValues_mean, historical_conversion_mean,
            historical_xi_mean, last_three_LeastWinningCost_mean, last_three_pValues_mean,
            last_three_conversion_mean, last_three_xi_mean,
            current_pValues_mean, current_pv_num, last_three_pv_num_total,
            historical_pv_num_total
        ])

        if timeStepIndex == 0:
            self.model.init_eval()
        
        # target_return = self.target_return
        # if timeStepIndex != 0 and timeStepIndex < 5:
        #     target_return = None
        # elif timeStepIndex != 0:
        #     cur_CPA = min(50000, self.budget * (1 - budget_left) / (sum([sum(i) for i in history_conversion]) + 1e-5))

        #     times = min(self.cpa / (cur_CPA + 1e-5), 4)
        #     if cur_CPA == 50000:
        #         times = 4
        #     target_return = times * self.model.eval_target_return[0, -1] - (sum(history_conversion[-1]) / self.scale)
        alpha = 10 * self.model.take_actions(test_state,
                                        pre_reward=sum(history_conversion[-1]) if len(history_conversion) != 0 else None, target_return=self.target_return)
        

        bids = alpha * pValues
        


        if budget_left - 1 / 48 > (48 - timeStepIndex - 1):
            bids = alpha * 1.5 * pValues


        return bids


