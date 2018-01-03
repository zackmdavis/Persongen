import itertools
import random
from enum import IntEnum

from sklearn.naive_bayes import GaussianNB
from numpy import array
from numpy.random import normal


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
        [Person.generate_female() for _ in range(size)] +
        [Person.generate_male() for _ in range(size)]
    )


def model_population(test, validation):
    model = GaussianNB()
    test_data = array([person.trait_vector() for person in test])
    test_target = array([int(person.sex) for person in test])
    model.fit(test_data, test_target)

    validation_data = array([person.trait_vector() for person in validation])
    validation_target = array([int(person.sex) for person in validation])

    predictions = model.predict(validation_data)

    hits = 0
    for person, predicted in zip(validation, predictions):
        print("Person traits: {}; predicted: {}; actual: {}".format(person.trait_vector(), predicted, person.sex))
        if predicted == person.sex:
            hits += 1

    print("total accuracy {}/{} = {}%".format(hits, len(validation), hits/len(validation)*100))


if __name__ == "__main__":
    print(Person.generate_male())
    model_population(simulate_population(10000), simulate_population(10000))
