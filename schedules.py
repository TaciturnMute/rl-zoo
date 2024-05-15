from typing import Dict
import matplotlib.pyplot as plt


class linear_schedule():
    def __init__(self, start_point: float, end_point: float, end_time: int):
        self.start_point = start_point
        self.end_point = end_point
        self.end_time = end_time
        self.current = start_point
        self.inc = (self.end_point - self.start_point) / self.end_time
        self.bound = min if end_point > start_point else max

    def _reset(self):
        self.current = self.start_point

    def __call__(self):
        value = self.bound(self.current + self.inc, self.end_point)
        self.current = value
        return value

    def Plot(self):
        self._reset()
        epsilons = []
        for i in range(int(self.end_time * 1.3)):
            epsilons.append(self.__call__())
        plt.figure(figsize=(10, 5))
        plt.plot(epsilons, label='linear_schedule')
        plt.legend()
        plt.show()

class inverse_ratio_schedule():

    def __init__(self, start_point: int, end_point: int, start_time: int, end_time: int):
        '''
        # declines with from start_point to end_point, between time start_time and time end_time
        :param start_point:
        :param end_point:
        :param start_time:
        :param end_time:
        '''
        self.start_point = start_point
        self.end_point = end_point
        self.start_time = start_time
        self.end_time = end_time
        self.current_time = 1
        self.ratio = 10  # the value is bigger, the curve is sharper


    def get_value(self):

        value = 1 / (1 + (self.current_time - self.start_time)*self.ratio/(self.end_time - self.start_time)
                     )
        self.current_time += 1
        if self.end_point < self.start_point and value < self.end_point:
            return self.end_point
        if self.end_point > self.start_point and value > self.end_point:
            return self.end_point
        return value

    def _reset(self):
        self.current_time = 1

    def Plot(self):
        self._reset()
        epsilons = []
        for i in range(self.end_time):
            epsilons.append(self.get_value())
        plt.figure(figsize=(10, 5))
        plt.plot(epsilons, label='inverse_ratio_schedule')
        plt.legend()
        plt.show()


def get_schedule(aliase: str, kwargs: Dict):
    schedule_aliases: Dict = {"linear": linear_schedule,
                              "reverse_ratio": inverse_ratio_schedule}
    if aliase is None:
        return None
    else:
        return schedule_aliases[aliase](**kwargs)



# epsilon_schedule_kwargs = {
#     'end_time': None,
#     'start_point': None,
#     'end_point': None,
#
# }
# epsilon_schedule_kwargs.update({'end_time':50000,
#                                'start_point':1,
#                                'end_point':0})
# epsilon_schedule = get_schedule('linear', epsilon_schedule_kwargs)
# epsilon_schedule.Plot()


# class ConstantSchedule:
#     def __init__(self, val):
#         self.val = val
#
#     def __call__(self, steps=1):
#         return self.val
#
#
# class LinearSchedule:
#     def __init__(self, start, end=None, steps=None):
#         if end is None:
#             end = start
#             steps = 1
#         self.inc = (end - start) / float(steps)
#         self.current = start
#         self.end = end
#         if end > start:
#             self.bound = min
#         else:
#             self.bound = max
#
#     def __call__(self, steps=1):
#         val = self.current
#         self.current = self.bound(self.current + self.inc * steps, self.end)
#         return val

