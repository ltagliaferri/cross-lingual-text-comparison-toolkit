"""
Regression tests for the two plotting helpers whose bugs only showed up
in the rendered figure: which terms get compared, and how the subplot
grid is unwrapped.
"""

from collections import Counter

from cross_lingual_toolkit.dependency_parsing import agent_grid
from cross_lingual_toolkit.metrics import top_shared_terms


# ---------------------------------------------------------------------------
# top_shared_terms()
# ---------------------------------------------------------------------------

def test_top_shared_terms_picks_by_count_not_insertion_order():
    # regression: the selection used to slice Counter.keys(), which is in
    # insertion order — so a verb first seen late in the corpus was
    # dropped from the chart no matter how frequent it was.
    a = Counter({f'rare{i}': 1 for i in range(20)})
    a['frequent'] = 999          # inserted last
    assert 'frequent' in top_shared_terms(a, Counter(), 12)


def test_top_shared_terms_ranks_by_combined_count():
    a = Counter({'x': 5, 'y': 1})
    b = Counter({'y': 10})
    assert top_shared_terms(a, b, 2) == ['y', 'x']


def test_top_shared_terms_unions_both_counters():
    a = Counter({'only_a': 1})
    b = Counter({'only_b': 1})
    assert set(top_shared_terms(a, b, 10)) == {'only_a', 'only_b'}


def test_top_shared_terms_respects_n():
    a = Counter({'a': 3, 'b': 2, 'c': 1})
    assert len(top_shared_terms(a, Counter(), 2)) == 2


def test_top_shared_terms_empty_counters_return_empty():
    assert top_shared_terms(Counter(), Counter(), 12) == []


# ---------------------------------------------------------------------------
# agent_grid()
# ---------------------------------------------------------------------------

def test_agent_grid_single_agent_yields_one_usable_axes():
    # regression: with one agent the old code built a 1x2 grid and then
    # wrapped the whole ndarray in a list, so the caller ended up calling
    # .set_title() on the array itself.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes_flat = agent_grid(1)
    assert len(axes_flat) == 1
    axes_flat[0].set_title('works')   # would raise on a bare ndarray
    plt.close(fig)


def test_agent_grid_gives_one_axes_per_agent():
    import matplotlib.pyplot as plt
    for n in (1, 2, 3, 4, 5):
        fig, axes_flat = agent_grid(n)
        assert len(axes_flat) >= n
        for ax in axes_flat:
            ax.set_xlabel('x')        # every cell must be a real Axes
        plt.close(fig)
