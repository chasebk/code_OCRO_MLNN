import numpy as np
import time
from model.root.root_base import RootBase
from utils.MathUtil import elu, relu, tanh, sigmoid
from sklearn.metrics import mean_absolute_error, mean_squared_error

class RootHybridMlnn(RootBase):
    """
        This is root of all hybrid models which include Multi-layer Neural Network and Optimization Algorithms.
    """
    def __init__(self, root_base_paras=None, root_hybrid_paras=None):
        RootBase.__init__(self, root_base_paras)
        self.epoch = root_hybrid_paras["epoch"]
        self.activations = root_hybrid_paras["activations"]
        self.hidden_size = root_hybrid_paras["hidden_size"]
        self.train_valid_rate = root_hybrid_paras["train_valid_rate"]
        self.domain_range = root_hybrid_paras["domain_range"]

        if self.activations[0] == 0:
            self._activation1__ = elu
        elif self.activations[0] == 1:
            self._activation1__ = relu
        elif self.activations[0] == 2:
            self._activation1__ = tanh
        else:
            self._activation1__ = sigmoid

        if self.activations[1] == 0:
            self._activation2__ = elu
        elif self.activations[1] == 1:
            self._activation2__ = relu
        elif self.activations[1] == 2:
            self._activation2__ = tanh
        else:
            self._activation2__ = sigmoid

    def _setting__(self):
        self.input_size, self.output_size = self.X_train.shape[1], self.y_train.shape[1]
        self.w1_size = self.input_size * self.hidden_size
        self.b1_size = self.hidden_size
        self.w2_size = self.hidden_size * self.output_size
        self.b2_size = self.output_size
        self.problem_size = self.w1_size + self.b1_size + self.w2_size + self.b2_size
        self.root_algo_paras = {
            "X_train": self.X_train, "y_train": self.y_train, "X_valid": self.X_valid, "y_valid": self.y_valid,
            "train_valid_rate": self.train_valid_rate, "domain_range": self.domain_range,
            "problem_size": self.problem_size, "print_train": self.print_train,
            "_get_average_error__": self._get_average_error__
        }

    def _forecasting__(self):
        hidd = self._activation1__(np.add(np.matmul(self.X_test, self.model["w1"]), self.model["b1"]))
        y_pred = self._activation2__(np.add(np.matmul(hidd, self.model["w2"]), self.model["b2"]))
        real_inverse = self.scaler.inverse_transform(self.y_test)
        pred_inverse = self.scaler.inverse_transform(np.reshape(y_pred, self.y_test.shape))
        return real_inverse, pred_inverse, self.y_test, y_pred

    def _running__(self):
        self.time_system = time.time()
        self._preprocessing_2d__()
        self._setting__()
        self.time_total_train = time.time()
        self._training__()
        self._get_model__(self.solution)
        self.time_total_train = round(time.time() - self.time_total_train, 4)
        self.time_epoch = round(self.time_total_train / self.epoch, 4)
        self.time_predict = time.time()
        y_actual, y_predict, y_actual_normalized, y_predict_normalized = self._forecasting__()
        self.time_predict = round(time.time() - self.time_predict, 6)
        self.time_system = round(time.time() - self.time_system, 4)
        if self.test_type == "normal":
            self._save_results__(y_actual, y_predict, y_actual_normalized, y_predict_normalized, self.loss_train)
        elif self.test_type == "stability":
            self._save_results_ntimes_run__(y_actual, y_predict, y_actual_normalized, y_predict_normalized)

    ## Helper functions
    def _get_model__(self, individual=None):
        w1 = np.reshape(individual[:self.w1_size], (self.input_size, self.hidden_size))
        b1 = np.reshape(individual[self.w1_size:self.w1_size + self.b1_size], (-1, self.hidden_size))
        w2 = np.reshape(individual[self.w1_size + self.b1_size: self.w1_size + self.b1_size + self.w2_size],
                        (self.hidden_size, self.output_size))
        b2 = np.reshape(individual[self.w1_size + self.b1_size + self.w2_size:], (-1, self.output_size))
        self.model = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

    def _get_average_error__(self, individual=None, X_data=None, y_data=None):
        w1 = np.reshape(individual[:self.w1_size], (self.input_size, self.hidden_size))
        b1 = np.reshape(individual[self.w1_size:self.w1_size + self.b1_size], (-1, self.hidden_size))
        w2 = np.reshape(individual[self.w1_size + self.b1_size: self.w1_size + self.b1_size + self.w2_size],
                        (self.hidden_size, self.output_size))
        b2 = np.reshape(individual[self.w1_size + self.b1_size + self.w2_size:], (-1, self.output_size))
        hidd = self._activation1__(np.add(np.matmul(X_data, w1), b1))
        y_pred = self._activation2__(np.add(np.matmul(hidd, w2), b2))
        return [mean_squared_error(y_pred, y_data), mean_absolute_error(y_pred, y_data)]
