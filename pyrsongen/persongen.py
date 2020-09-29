import itertools
import random
from enum import IntEnum

import matplotlib.pyplot as plot
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from numpy import array
from numpy.random import normal
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA


Sex = IntEnum('Sex', [
    'female', # ♀
    'male', # ♂
])

class Person:
    def __init__(self, sex, agreeableness, neuroticism, enthusiasm, assertiveness, industriousness, orderliness, people_orientation, visuospatial, verbal, risk_taking, sociosexuality):
        self.sex = sex

        # TODO: plot personality–interest-ability–audacity instead of using PCA

        self.agreeableness = agreeableness
        self.neuroticism = neuroticism

        # extraversion facets
        self.enthusiasm = enthusiasm
        self.assertiveness = assertiveness

        # conscientiousness facets
        self.industriousness = industriousness
        self.orderliness = orderliness

        self.people_orientation = people_orientation
        self.visuospatial = visuospatial
        self.verbal = verbal

        self.risk_taking = risk_taking
        self.sociosexuality = sociosexuality

    def trait_vector(self):
        return [self.agreeableness, self.neuroticism,
                self.enthusiasm, self.assertiveness, self.industriousness, self.orderliness,
                self.people_orientation,
                self.visuospatial, self.verbal, self.risk_taking, self.sociosexuality]

    # We'll assume these normally-distributed sex differences:
    #
    # • Agreeableness, d=0.48
    # • Neuroticism, d=0.39
    # • Enthusiasm, d=0.32
    # • Assertiveness, d=0.24
    # • Industriousness, d=0.15
    # • Orderliness, d=0.22
    # • People–Things, d=0.93
    # • Visuospatial ability d=0.48
    # • Verbal ability d=0.27
    # • Risk-taking d=0.49
    # • Sociosexuality d=0.74
    #
    # See Weisberg et al. "Gender Differences in Personality across the Ten
    # Aspects of the Big Five" (_Frontiers in Psychology_,
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3149680/), Su et al. "Men
    # and Things, Women and People: A Meta-Analysis of Sex Differences in
    # Interests" (_Psychological Bulletin_), and Archer, "The Reality and
    # Evolutionary Significance of Human Psychological Sex Differences"

    @classmethod
    def generate_female(cls):
        agreeableness = normal(0.24, 1)
        neuroticism = normal(0.195, 1)
        enthusiasm = normal(0.16, 1)
        assertiveness = normal(-0.12, 1)
        industriousness = normal(-0.075, 1)
        orderliness = normal(0.11, 1)
        people_orientation = normal(0.465, 1)
        visuospatial = normal(-0.24, 1)
        verbal = normal(0.135, 1)
        risk_taking = normal(-0.245, 1)
        sociosexuality = normal(-0.37, 1)
        return cls(Sex.female, agreeableness, neuroticism, enthusiasm, assertiveness, industriousness, orderliness, people_orientation, visuospatial, verbal, risk_taking, sociosexuality)

    @classmethod
    def generate_male(cls):
        agreeableness = normal(-0.24, 1)
        neuroticism = normal(-0.195, 1)
        enthusiasm = normal(-0.16, 1)
        assertiveness = normal(0.12, 1)
        industriousness = normal(0.075, 1)
        orderliness = normal(-0.11, 1)
        people_orientation = normal(-0.465, 1)
        visuospatial = normal(0.24, 1)
        verbal = normal(-0.135, 1)
        risk_taking = normal(0.245, 1)
        sociosexuality = normal(0.37, 1)
        return cls(Sex.male, agreeableness, neuroticism, enthusiasm, assertiveness, industriousness, orderliness, people_orientation, visuospatial, verbal, risk_taking, sociosexuality)


def simulate_population(size):
    return (
        [Person.generate_female() for _ in range(size//2)] +
        [Person.generate_male() for _ in range(size//2)]
    )


def data_array(population):
    return array([person.trait_vector() for person in population])


def target_array(population):
    return array([int(person.sex) for person in population])


def fit_model(train):
    model = GaussianNB()
    train_data = data_array(train)
    train_target = target_array(train)
    model.fit(train_data, train_target)
    return model


def model_population(train, test):
    model = fit_model(train)

    test_data = array([person.trait_vector() for person in test])
    test_target = array([int(person.sex) for person in test])

    point_predictions = model.predict(test_data)
    log_prob_predictions = model.predict_log_proba(test_data)
    prob_predictions = model.predict_proba(test_data)

    print("score: {}".format(model.score(test_data, test_target)))

    hits = 0
    bits = 0
    for person, predicted, log_probs, probs in zip(test, point_predictions, log_prob_predictions, prob_predictions):
        # print("Person traits: {}; predicted: {}; actual: {}; probabilities: {}".format(person.trait_vector(), predicted, person.sex, probs))
        if predicted == person.sex:
            hits += 1

        bits += log_probs[person.sex-1]

    print("total accuracy {}/{} = {}%".format(hits, len(test), hits/len(test)*100))
    print("Bayes-score {} bits".format(bits))


def plot_data(data, target):
    figure = plot.figure(figsize=(6, 5))
    axes = Axes3D(figure)

    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(data)
    print(pca.components_)

    axes.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
                 c=target, cmap=ListedColormap(["#FF1493", "#1E90FF"]))

    # axes.set_xlabel("Agreeableness")
    # axes.set_ylabel("Neuroticism")
    # axes.set_zlabel("people–things orientation")

    plot.show()


if __name__ == "__main__":
    train, test = simulate_population(5000), simulate_population(5000)
    model_population(train, test)

    plot_population = simulate_population(300)
    plot_data(data_array(plot_population), target_array(plot_population))

    # train = simulate_population(5000)
    # model = fit_model(train)
    # import IPython
    # IPython.embed()


# In [8]: synth_model = GaussianNB()

# In [9]: from numpy import array

# In [10]: synth_model.class_prior_ = array([0.5, 0.5])

# In [11]: synth_model.classes_ = array([1, 2])

# In [13]: synth_model.sigma_ = array([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]])

# In [14]: synth_model.theta = array([[0.24, 0.195, 0.465, -0.24, 0.135, -0.245, -0.37], [-0.24, -0.19
#     ...: 5, -0.465, 0.24, -0.135, 0.245, 0.37]])
