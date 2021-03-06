from collections import OrderedDict

import numpy as np


class LogBuffer(object):

    def __init__(self):
        self.val_history = OrderedDict()
        self.n_history = OrderedDict()
        self.output = OrderedDict()
        self.ready = False

    def clear(self):
        self.val_history.clear()
        self.n_history.clear()
        self.clear_output()

    def clear_output(self):
        self.output.clear()
        self.ready = False

    def update(self, vars, count=1):
        assert isinstance(vars, dict)
        for key, var in vars.items():
            if key not in self.val_history:
                self.val_history[key] = []
                self.n_history[key] = []
            self.val_history[key].append(var)
            self.n_history[key].append(count)

    def average(self, n=0):
        """Average latest n values or all values"""
        assert n >= 0
        for key in self.val_history:
            values = np.array(self.val_history[key][-n:])
            nums = np.array(self.n_history[key][-n:])
            avg = np.sum(values * nums) / np.sum(nums)
            self.output[key] = avg
        self.ready = True


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



# from collections import OrderedDict
#
# import numpy as np
#
#
# class LogBuffer(object):
#
#     def __init__(self):
#         self.val_history = OrderedDict()
#         self.n_history = OrderedDict()
#         self.epoch_history = OrderedDict()
#         self.n_epoch_history = 0
#         self.output = OrderedDict()
#         self.ready = False
#
#     def clear(self):
#         self.val_history.clear()
#         self.n_history.clear()
#         self.clear_output()
#
#     def clear_epoch(self):
#         self.epoch_history.clear()
#         self.n_epoch_history = 0
#
#
#     def clear_output(self):
#         self.output.clear()
#         self.ready = False
#
#     def update(self, vars, count=1):
#         assert isinstance(vars, dict)
#         for key, var in vars.items():
#             if('acc' not in key):
#                 if key not in self.val_history:
#                     self.val_history[key] = []
#                     self.n_history[key] = []
#                 self.val_history[key].append(var)
#                 self.n_history[key].append(count)
#             else:
#                 if key not in self.epoch_history:
#                     self.epoch_history[key] = []
#                 self.epoch_history[key].append(var)
#                 self.n_epoch_history += 1
#
#
#     def average(self, n=0):
#         """Average latest n values or all values"""
#         assert n >= 0
#         for key in self.val_history:
#             values = np.array(self.val_history[key][-n:])
#             nums = np.array(self.n_history[key][-n:])
#             avg = np.sum(values * nums) / np.sum(nums)
#             self.output[key] = avg
#         for key in self.epoch_history:
#             self.output[key] = np.mean(self.epoch_history[key])
#
#         self.ready = True