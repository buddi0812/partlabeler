"""Smart sorting (engine/sort.py) on synthetic features: three kinds of image, a few duplicates, one wrong label."""
import numpy as np

from engine.sort import Groups, near_duplicates, odd_ones, predict


def blobs(n=40, dims=64, spread=0.35, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(3, dims))
    v = np.concatenate([c + spread * rng.normal(size=(n, dims)) for c in centres])
    return v, np.repeat([0, 1, 2], n)


def test_groups_find_the_kinds_and_the_slider_recuts():
    v, truth = blobs()
    g = Groups(v)
    assert g.best_k == 3
    cut = g.cut()
    assert sorted(len(x) for x in cut["groups"]) == [40, 40, 40]
    assert all(len(set(truth[x])) == 1 for x in cut["groups"]) and cut["unsure"] == []
    assert g.cut(5)["k"] == 5 and g.cut(1)["groups"] == [list(g.cut(1)["groups"][0])]


def test_near_duplicates_are_stacked():
    v, _ = blobs()
    v = np.concatenate([v, v[[3, 50]] + 1e-4])               # items 120, 121 copy items 3 and 50
    assert near_duplicates(v) == [[3, 120], [50, 121]]


def test_a_few_examples_suggest_the_rest():
    v, truth = blobs()
    few = {0: 0, 1: 0, 40: 1, 41: 1, 80: 2, 81: 2}               # prototypes
    cls, score, fits = predict(v, few)
    assert (cls == truth).mean() > 0.97 and score.min() > 0.3
    two = {0: 0, 1: 0, 40: 1, 41: 1}                              # nobody has named kind 2 yet: no guesses for it
    fits = predict(v, two)[2]
    assert fits[80:].mean() < 0.1 and fits[:80].mean() > 0.3        # S10 on real crops: 46-72% suggested
    many = {i: int(truth[i]) for i in range(0, 120, 4)}           # logistic regression from 5 per class up
    assert (predict(v, many)[0] == truth).mean() > 0.97
    assert predict(v, {0: 0, 1: 0}) is None


def test_odd_ones_find_the_wrong_label():
    v, truth = blobs()
    labels = {i: int(truth[i]) for i in range(120)}
    labels[7] = 2                                                 # a kind-0 item labeled as kind 2
    odd = odd_ones(v, labels)
    assert odd[0][:2] == (7, 0) and len(odd) == 1
