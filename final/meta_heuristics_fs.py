"""
@author: Md Azimul Haque
"""
import time
import warnings
import sys
import os

from copy import deepcopy

import numpy as np
import pandas as pd


if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


class AntColonyOptimizationFS:
    """
    Machine Learning Parameters
    ----------

    columns_list : Column names present in x_train_dataframe and x_test which will be used as input list for searching best list of features.

    data_dict : X and Y training and test data provided in dictionary format. Below is example of 5 fold cross validation data with keys.
        {0:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array},
        1:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array},
        2:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array},
        3:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array},
        4:{'x_train':x_train_dataframe,'y_train':y_train_array,'x_test':x_test_dataframe,'y_test':y_test_array}}

        If you only have train and test data and do not wish to do cross validation, use above dictionary format, with only one key.

    use_validation_data : Whether you want to use validation data as a boolean True or False. Default value is True. If false, user need not provide x_validation_dataframe and y_validation_dataframe

    x_validation_dataframe : dataframe containing features of validatoin dataset

    y_validation_dataframe : dataframe containing dependent variable of validation dataset

    model : Model object. It should have .fit and .predict attribute

    cost_function_improvement : Objective is to whether increase or decrease the cost during subsequent iterations.
        For regression it should be 'decrease' and for classification it should be 'increase'

    cost_function : Cost function for finding cost between actual and predicted values, depending on regression or classification problem.
        cost function should accept 'actual' and 'predicted' as arrays and return cost for the both.

    average : Averaging to be used. This is useful for clasification metrics such as 'f1_score', 'jaccard_score', 'fbeta_score', 'precision_score',
        'recall_score' and 'roc_auc_score' when dependent variable is multi-class

    Ant Colony Optimization Parameters
    ----------

    iterations : Number of times ant colony optimization will search for solutions. Default is 100

    N_ants : Number of ants in each iteration. Default is 100.

    run_time : Number of minutes to run the algorithm. This is checked in between each iteration.
        At start of each generation it is checked if runtime has exceeded than alloted time.
        If case run time did exceeds provided limit, best result from iterations executed so far is given as output.
        Default is 2 hours. i.e. 120 minutes.

    evaporation_rate : Evaporation rate. Values are between 0 and 1. If it is too large, chances are higher to find global optima, but computationally expensive. If it is low, chances of finding global optima are less. Default is kept as 0.9

    Q : Pheromene update coefficient. Value between 0 and 1. It affects the convergence speed. If it is large, ACO will get stuck at local optima. Default is kept as 0.2

    Output
    ----------
    best_columns : List object with list of column names which gives best performance for the model. These features can be used for training and saving models separately by the user.

    """

    def __init__(
        self,
        columns_list,
        data_dict,
        model,
        cost_function,
        x_validation_dataframe=pd.DataFrame(),
        y_validation_dataframe=pd.DataFrame(),
        use_validation_data=True,
        penalize_parameters_size=True,
        cost_function_improvement="increase",
        average=None,
        iterations=100,
        n_ants=100,
        run_time=120,
        print_iteration_result=True,
        evaporation_rate=0.9,
        Q=0.2,
    ):
        self.columns_list = columns_list
        self.data_dict = data_dict
        self.x_validation_dataframe = x_validation_dataframe
        self.y_validation_dataframe = y_validation_dataframe
        self.use_validation_data = use_validation_data
        self.model = model
        self.cost_function = cost_function
        self.cost_function_improvement = cost_function_improvement
        self.average = average
        self.iterations = iterations
        self.n_ants = n_ants
        self.run_time = run_time
        self.print_iteration_result = print_iteration_result
        self.evaporation_rate = evaporation_rate
        self.penalize_parameters_size = penalize_parameters_size
        self.Q = Q

        self.last_score = 0
        self.repeated_count = 0
        self.fp = [1] * (len(columns_list))
        self.ants: list[Ant] = []
        self.size = len(columns_list)
        self.top_score = 0
        self.result = []

    def _calculate_cost(self, current_at_feature_subset):
        fold_cost = []

        for i in self.data_dict.keys():
            x_train = self.data_dict[i]["x_train"][current_at_feature_subset]
            y_train = self.data_dict[i]["y_train"]

            x_test = self.data_dict[i]["x_test"][current_at_feature_subset]
            y_test = self.data_dict[i]["y_test"]

            self.model.fit(x_train, y_train)

            y_test_predict = self.model.predict(x_test)

            if self.use_validation_data:
                y_validation_predict = self.model.predict(
                    self.x_validation_dataframe[current_at_feature_subset]
                )

            if self.average:
                fold_cost.append(
                    self.cost_function(
                        y_test, y_test_predict, average=self.average
                    )
                )
            if self.use_validation_data:
                fold_cost.append(
                    self.cost_function(
                        self.y_validation_dataframe,
                        y_validation_predict,
                        average=self.average,
                    )
                )
            elif self.penalize_parameters_size:
                fold_cost.append(
                    self.cost_function(
                        y_test,
                        y_test_predict,
                        current_params_size=len(current_at_feature_subset),
                        total_params_size=len(self.columns_list)
                    )
                )

            else:
                fold_cost.append(self.cost_function(y_test, y_test_predict))
                if self.use_validation_data:
                    fold_cost.append(
                        self.cost_function(
                            self.y_validation_dataframe, y_validation_predict
                        )
                    )

        return np.mean(fold_cost)

    def _constructAntSolution(self, ant):
        current_at_feature_subset = []

        featureSetIndex = []
        # for each feature index
        for j in range(self.size):
            # generate random number
            decision = np.random.rand()

            # in the first iteration, fp has all values as 1 for the length of features

            if decision < self.fp[j] / 2.0:
                featureSetIndex.append(1)
                current_at_feature_subset.append(self.columns_list[j])
            else:
                featureSetIndex.append(0)

        # if no features, then just keep 0.5. Else, calculate the actual cost
        if sum(featureSetIndex) == 0:
            score = 0.5
        else:
            score = float(self._calculate_cost(current_at_feature_subset))

        # for the ant, assign score and feature indexes used.
        ant.val = score
        ant.subsets = deepcopy(featureSetIndex)

        return ant

    def _ApplyLocalSearch(self):
        maxSet = []

        if self.cost_function_improvement == "decrease":
            maxScore = np.inf
        else:
            maxScore = 0

        # for each ant in the iteration
        for a in self.ants:
            if self.cost_function_improvement == "decrease":
                if maxScore > a.val or (
                    maxScore == a.val and (
                        maxSet and sum(a.subsets) < sum(maxSet))
                ):
                    maxScore = a.val
                    maxSet = a.subsets
            else:
                if maxScore < a.val or (
                    maxScore == a.val and (
                        maxSet and sum(a.subsets) < sum(maxSet))
                ):
                    maxScore = a.val
                    maxSet = a.subsets

        # After the search for best score is done and associated feature set binary vector is found,

        if self.cost_function_improvement == "decrease":
            if self.top_score > maxScore or (
                maxScore == self.top_score
                and (self.result and sum(maxSet) < sum(self.result))
            ):
                self.top_score = maxScore
                self.result = maxSet
        else:
            if self.top_score < maxScore or (
                maxScore == self.top_score
                and (self.result and sum(maxSet) < sum(self.result))
            ):
                self.top_score = maxScore
                self.result = maxSet

        # but return only local best result for current colony
        return maxSet, maxScore

    def _calc_update_param(self, top_score):
        sumResults = 0

        for a in self.ants:
            if sum(a.subsets) > 0:
                # value that is added is pehermone update coefficient divided by cost, for both current and top ant
                sumResults += self.Q / a.val

        return sumResults + (self.Q / top_score)

    def _UpdatePheromoneTrail(self, top_set, top_score):
        # get sum results
        sumResults = self._calc_update_param(top_score)

        # top_set is binary 1|0 feature vector. top_score is best score for entire colony

        for i, v in enumerate(top_set):
            # evaporate pheromene, based on formula 𝜏𝑖 = 𝜌 ∗ 𝜏𝑖

            pheromone_at_index = self.fp[i] * self.evaporation_rate

            # update pheromene trail

            if v == 1:
                pheromone_at_index += self.fp[i] + sumResults

            self.fp[i] = pheromone_at_index

    def GetBestFeatures(self):
        if self.cost_function_improvement == "decrease":
            self.top_score = np.inf

        # get starting time
        start = time.time()

        # For each iteratoin of ACO
        for iter_num in range(self.iterations):
            print(f"Running iteration {iter_num}")
            # check if time exceeded
            if (time.time() - start) // 60 > self.run_time:
                print(
                    "================= Run time exceeded allocated time. Producing best solution generated so far. ================="
                )
                break

            # for each ant
            for _ in range(self.n_ants):
                # create new ant
                ant = Ant()
                # create the first initialization for ant
                ant = self._constructAntSolution(ant)
                self.ants.append(ant)

            # for the iteration, after all colony of ants have been created

            top_set, top_score = self._ApplyLocalSearch()

            if self.top_score == self.last_score:
                self.repeated_count += 1

            if self.repeated_count > 5:
                print(
                    "================= No improvement in last 5 iterations. Stopping the search. ================="
                )
                break

            self.last_score = self.top_score

            if self.print_iteration_result:
                print(
                    "Best combined performance on test and validation data for iteration "
                    + str(iter_num)
                    + ": "
                    + str(self.top_score)
                )

            # give input the best feature binary 1|0 vector and best metric from the entire colony
            self._UpdatePheromoneTrail(top_set, top_score)
            self.ants = []

        # After everything is done, just get the original name of feature.
        best_columns = []
        for indx, _ in enumerate(self.result):
            if self.result[indx] == 1:
                best_columns.append(self.columns_list[indx])

        print("================= Best result:",
              self.top_score, "=================")
        print(
            "================= Execution time in minutes:",
            (time.time() - start) // 60,
            "=================",
        )

        return best_columns


class Ant:
    def __init__(self):
        self.subsets = []
        self.val = 0
