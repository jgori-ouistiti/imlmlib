import numpy
from abc import ABC, abstractmethod


class MemoryModel:
    def __init__(self, nitems, *args, seed=None, **kwargs):
        self.nitems = nitems
        self._original_seed = seed
        self.rng = numpy.random.default_rng(seed=seed)

    @abstractmethod
    def update(self, item, time):
        raise NotImplementedError

    @abstractmethod
    def compute_probabilities(self, time=None):
        raise NotImplementedError

    @property
    def seed(self):
        if getattr(self, "_seed", None) is None:
            return self._original_seed
        else:
            return self._seed

    def reset(self):
        if isinstance(self.seed, numpy.random.SeedSequence):
            self._seed = self._original_seed.spawn(1)[0]
        return

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
        query = memory_model.query_item(item, time)
        queries.append(query)
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

    n = 0
    for memory_model in population_model:
        for i in range(replications):
            trial_data = numpy.array(trial(memory_model, schedule)[0])
            data[i, :, :, n] = trial_data.T
        n += 1
    return data


def serialize_experiment(data):
    _shape = data.shape
    data = data.reshape(_shape[0], _shape[1], -1, order="F")  # careful for the order
    k_vector = []
    for i in range(_shape[3]):
        k_vector += [k for k in range(-1, _shape[2] - 1)]
    return data, k_vector


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
            ret_value = self[self.counter]
            self.counter += 1
            return ret_value
        else:
            raise StopIteration

    def __repr__(self):
        return f"items: {self.items}\ntimes: {self.times}"


class BlockBasedSchedule(Schedule):
    def __init__(self, nitems, intertrial_time, interblock_time, repet_trials=2, seed = None, sigma_t = None):

        self.seed = seed
        self.rng = numpy.random.default_rng(self.seed)
        self.sigma_t = sigma_t

        self.nitems = nitems
        self.intertrial_time = intertrial_time
        self.interblock_time = interblock_time
        self.repet_trials = repet_trials if repet_trials is not None else 1

        self.items, self.times, self.blocks = self.regen_schedule()
        self.max = numpy.array(self.times).squeeze().shape[0]
        

    def regen_schedule(
        self, intertrial_time=None, interblock_time=None, repet_trials=None
    ):
        self.intertrial_time = (
            self.intertrial_time if intertrial_time is None else intertrial_time
        )
        self.interblock_time = (
            self.interblock_time if interblock_time is None else interblock_time
        )
        self.repet_trials = self.repet_trials if repet_trials is None else repet_trials

        items = []
        times = []
        blocks = []
        t = 0
        for nb, ibt in enumerate((self.interblock_time + [0])):
            for r in range(self.repet_trials):
                if self.sigma_t is not None:
                    _rdm = self.rng.lognormal(sigma = self.sigma_t, size = (self.nitems,))
                else:
                    _rdm = numpy.zeros((self.nitems,))
                for i,r in zip(range(self.nitems), _rdm):
                    items.append(i)
                    times.append(t)
                    
                    t += self.intertrial_time + r
                    blocks.append(nb)
            t += ibt

        return items, times, blocks
    

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
