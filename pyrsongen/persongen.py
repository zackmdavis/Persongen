import itertools
import random
from enum import IntEnum

import matplotlib.pyplot as plot
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from numpy import array
from numpy.random import normal
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


Sex = IntEnum('Sex', [
    'female', # ♀
    'male', # ♂
])

class Person:
    def __init__(self, sex, agreeableness, neuroticism, people_orientation):
        self.sex = sex
        self.agreeableness = agreeableness
        self.neuroticism = neuroticism
        self.people_orientation = people_orientation

    def trait_vector(self):
        return [self.agreeableness, self.neuroticism, self.people_orientation]

    # We'll assume these normally-distributed sex differences:
    #
    # • Agreeableness, d=0.48
    # • Neuroticism, d=0.39
    # • People–Things, d=0.93
    #
    # See Weisberg et al. "Gender Differences in Personality across the Ten
    # Aspects of the Big Five" (_Frontiers in Psychology_,
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3149680/) and Su et al. "Men
    # and Things, Women and People: A Meta-Analysis of Sex Differences in
    # Interests" (_Psychological Bulletin_)

    @classmethod
    def generate_female(cls):
        agreeableness = normal(0.24, 1)
        neuroticism = normal(0.195, 1)
        people_orientation = normal(0.465, 1)
        return cls(Sex.female, agreeableness, neuroticism, people_orientation)

    @classmethod
    def generate_male(cls):
        agreeableness = normal(-0.24, 1)
        neuroticism = normal(-0.195, 1)
        people_orientation = normal(-0.465, 1)
        return cls(Sex.male, agreeableness, neuroticism, people_orientation)


def simulate_population(size):
    return (
        [Person.generate_female() for _ in range(size//2)] +
        [Person.generate_male() for _ in range(size//2)]
    )


def data_array(population):
    return array([person.trait_vector() for person in population])


def target_array(population):
    return array([int(person.sex) for person in population])


def model_population(test, validation):
    model = GaussianNB()
    test_data = data_array(test)
    test_target = target_array(test)
    model.fit(test_data, test_target)

    validation_data = array([person.trait_vector() for person in validation])
    validation_target = array([int(person.sex) for person in validation])

    point_predictions = model.predict(validation_data)
    log_prob_predictions = model.predict_log_proba(validation_data)
    prob_predictions = model.predict_proba(validation_data)

    hits = 0
    bits = 0
    for person, predicted, log_probs, probs in zip(validation, point_predictions, log_prob_predictions, prob_predictions):
        print("Person traits: {}; predicted: {}; actual: {}; probabilities: {}".format(person.trait_vector(), predicted, person.sex, probs))
        if predicted == person.sex:
            hits += 1

        bits += log_probs[person.sex-1]

    print("total accuracy {}/{} = {}%".format(hits, len(validation), hits/len(validation)*100))
    print("Bayes-score {} bits".format(bits))


def plot_data(data, target):
    figure = plot.figure(figsize=(6, 5))
    axes = Axes3D(figure)

    axes.scatter(data[:, 0], data[:, 1], data[:, 2],
                 c=target, cmap=ListedColormap(["#FF1493", "#1E90FF"]))

    axes.set_xlabel("Agreeableness")
    axes.set_ylabel("Neuroticism")
    axes.set_zlabel("people–things orientation")

    plot.show()


if __name__ == "__main__":
    test, validation = simulate_population(5000), simulate_population(5000)
    model_population(test, validation)

    plot_population = simulate_population(300)
    plot_data(data_array(plot_population), target_array(plot_population))
