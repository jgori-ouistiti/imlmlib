import numpy
from abc import ABC, abstractmethod


class MemoryModel:
    def __init__(self, nitems, *args, seed=None, **kwargs):
        self.nitems = nitems
        self.rng = numpy.random.default_rng(seed=seed)

    @abstractmethod
    def update(self, item, time):
        raise NotImplementedError

    @abstractmethod
    def compute_probabilities(self, time=None):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    def query_item(self, item, time):
        prob = self.compute_probabilities(time=time)[item]
        return (self.rng.random() < prob, prob)


def trial(memory_model, schedule, reset=True):
    """trial applies a schedule to a memory model and gathers queries


    :param memory_model: The memory model that will be queried from
    :type memory_model: inherits from MemoryModel
    :param schedule: schedule of interrogations
    :type schedule: inherits from Schedule
    :return: (queries, memory_model), where queries = [(good recall?, recall_probability)]
    :rtype: list(tuple(boolean, float))
    """
    queries = []
    if reset:
        memory_model.reset()
    for item, time in schedule:
        queries.append(memory_model.query_item(item, time))
        memory_model.update(item, time)
    return queries, memory_model


def experiment(population_model, schedule, replications=1):
    """experiment run a memory experiment with a population model, a fixed schedule, and a set amount of replications per participant

    Returns simulated data, with an array of shape (replication, 2, len(schedule), population size). The second dimension holds (recall?, recall probability)

    :param population_model: an object that can be iterated over, where each item inherits from MemoryModel
    :type population_model:
    :param schedule: _description_
    :type schedule: inherits from an imlmlib Schedule
    :param replications: number of replications for each participant, defaults to 1
    :type replications: int, optional
    :return: the simulated data
    :rtype: numpy.array((replication, 2, len(schedule), population size))
    """
    data = numpy.zeros((replications, 2, len(schedule), population_model.pop_size))
    for i in range(replications):
        for n, memory_model in enumerate(population_model):
            trial_data = numpy.array(trial(memory_model, schedule)[0])
            data[i, :, :, n] = trial_data.T
    return data


class Schedule:
    """Iterate over (items, times) pairs.

    A schedule is an association between presented items and the times at which these where presented. One can iterate over a schedule to get a pair (item, time).
    """

    def __init__(self, items, times):
        """__init__ _summary_

        _extended_summary_

        :param items: item identifier
        :type items: numpy array_like
        :param times: timestamp at which item was presented to participant
        :type times: numpy array_like
        """
        self.items = items
        self.times = times
        self.max = numpy.array(times).squeeze().shape[0]

    def __len__(self):
        return self.max

    def __getitem__(self, item):
        return (self.items[item], self.times[item])

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        if self.counter < self.max:
            ret_value = (self.items[self.counter], self.times[self.counter])
            self.counter += 1
            return ret_value
        else:
            raise StopIteration

    def __repr__(self):
        return f"items: {self.items}\ntimes: {self.times}"


class Player:
    def __init__(self, schedule, memory_model):
        self._schedule = schedule
        self.memory_model = memory_model
        self.step_counter = 0

    def reset(self):
        self.memory_model.reset()
        self.step_counter = 0
        self.schedule = iter(self._schedule)

    def step(self, item=True, time=None):
        if item:
            self.memory_model.update(*next(self.schedule))
            return self.memory_model.compute_probabilities()
        else:
            return self.memory_model.compute_probabilities(time=time)

    def __iter__(self):
        return self

    def __next__(self):
        return self.step(item=True)

    def __repr__(self):
        return f"Schedule: {self.schedule.__repr__()}\n Model: {self.memory_model.__repr__()}"

    def _print_info(self):
        return self.memory_model._print_info()


class MLEPlayer(Player):
    def __init__(self, *args, **kwargs):
        self.ll = 0

    def step(self, recall, item, time):
        self.memory_model.compute_probabilities(item=item, time=time)
        self.actr_log_likelihood_sample(recall, *next(self.schedule))
        self.memory_model.update(*next(self.schedule))


if __name__ == "__main__":
    items = [0, 1, 0, 1, 0, 1, 0, 0]
    times = [0, 100, 126, 200, 252, 500, 4844, 5877]

    schedule = Schedule(items, times)

    # TEST = "PAVLIK2005"
    TEST = "EF"

    if TEST == "PAVLIK2005":
        # testing ACTR_Pavlik_2005
        memory_model = ACTR_Pavlik2005(
            2, a=0.177, c=0.217, s=0.25, tau=-0.7, buffer_size=6
        )

        player = Player(schedule, memory_model)
        player.reset()

        for n, prob in enumerate(player):
            print(prob)

    if TEST == "EF":
        memory_model = ExponentialForgetting(2, a=1e-3, b=0.5)
        player = Player(schedule, memory_model)
        player.reset()

        for n, prob in enumerate(player):
            print(prob)


class ACTR_Pavlik2005(MemoryModel):
    """ACTR_Pavlik2005

    Pavlik Jr, P. I., & Anderson, J. R. (2005). Practice and forgetting effects on vocabulary memory: An activation‐based model of the spacing effect. Cognitive science, 29(4), 559-586.

    https://onlinelibrary.wiley.com/doi/pdf/10.1207/s15516709cog0000_14"""

    def __init__(self, nitems, a=0.177, c=0.217, s=0.25, tau=-0.7, buffer_size=256):
        super().__init__(nitems)
        self.a, self.c, self.s, self.tau = a, c, s, tau
        self.buffer_size = buffer_size
        self.reset()

    def reset(self, a=None, c=None, s=None, tau=None):
        if a is not None:
            self.a = a
        if c is not None:
            self.c = c
        if s is not None:
            self.s = s
        if tau is not None:
            self.tau = tau

        self.counter_pres = numpy.zeros((self.nitems,))
        self.counter_d = numpy.zeros((self.nitems, self.buffer_size))
        self.counter_time = numpy.full((self.nitems, self.buffer_size), -numpy.inf)
        self.counter_time[:, 0] = self.a
        self.activation = numpy.full((self.nitems, self.buffer_size), -numpy.inf)

    def _print_info(self):
        print(f"n: \t {self.counter_pres}")
        print(f"d: \t {self.counter_d}")
        print(f"time: \t {self.counter_time}")
        print(f"activation: \t {self.activation}")

    def update(self, item, time):
        n = numpy.asarray(self.counter_pres[item], dtype=numpy.int32)
        self.counter_time[item, n] = time

        self.activation[item, n] = numpy.log(
            numpy.sum(
                numpy.power(
                    time - self.counter_time[item, :n], -self.counter_d[item, 1 : n + 1]
                )
            )
        )

        self.counter_pres[item] += 1
        n += 1
        self.counter_d[item, n] = (
            self.c * numpy.exp(self.activation[item, n - 1]) + self.a
        )

    def compute_probabilities(self, time=None):
        if time is None:
            time = numpy.max(self.counter_time)

        n = numpy.asarray(numpy.max(self.counter_pres), dtype=numpy.int32)

        # activation:
        # ti**-di
        signals = numpy.power(
            time - self.counter_time[:, :n], -self.counter_d[:, 1 : n + 1]
        )

        signals_with_nan = numpy.where(
            self.counter_time[:, :n] == -numpy.inf, numpy.nan, signals
        )

        activation = numpy.log(
            numpy.sum(
                numpy.nan_to_num(signals_with_nan, posinf=numpy.inf),
                axis=1,
            )
        )
        return 1 / (1 + numpy.exp((self.tau - activation) / self.s))

    def actr_log_likelihood_sample(self, recall, time, item):
        prob = self.compute_probabilities(time)
        self.update(item, time)
        if recall == 1:
            return numpy.log(prob)
        else:
            return numpy.log(1 - prob)

    def __repr__(self):
        return f"{self.__class__.__name__}\n a = {self.a}\n c = {self.c}\n tau = {self.tau}\n s = {self.s}"
