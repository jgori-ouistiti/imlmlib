from imlmlib.mem_utils import trial
from imlmlib.exponential_forgetting import ExponentialForgetting
from imlmlib.mem_utils import Schedule


def test_trial():
    ef_model = ExponentialForgetting(1)
    sched = Schedule(
        [0 for i in range(10)], [125, 250, 375, 500, 1000, 1500, 2000, 2500, 3000, 4000]
    )
    queries, memory_model = trial(ef_model, sched)
    assert queries[0] == (False, 0.0)  # first query is unknown by default


if __name__ == "__main__":
    test_trial()
