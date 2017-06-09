# requires Python 3.4+ (or enum backport), attrs
# (http://www.attrs.org/en/stable/), sckikit-learn, matplotlib

import random
from enum import Enum

import attr
import matplotlib.pyplot as plot
import numpy

from mpl_toolkits.mplot3d import Axes3D
from numpy.random import normal
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

Sex = Enum('Sex', [
    'female', # ♀
    'male', # ♂
])

class GenderBinaryException(ValueError):
    pass

def encode_sex(sex):
    if sex == Sex.female:
        return 0
    elif sex == Sex.male:
        return 1
    else:
        raise GenderBinaryException()

@attr.s
class TransWoman:
    attraction_before_srs = attr.ib()
    attraction_after_srs = attr.ib()
    age_at_srs = attr.ib()
    age_at_first_wish = attr.ib()
    age_at_full_time = attr.ib()
    duration_of_rlt = attr.ib()
    pre_srs_female_partners = attr.ib()
    pre_srs_male_partners = attr.ib()
    femme_child = attr.ib()
    femme_child_to_others = attr.ib()
    agp_hundreds = attr.ib()
    married_before_srs = attr.ib()
    sire_before_srs = attr.ib()

    def to_data_row(self):
        # Note: excluding pre-SRS partners, marriage, sire.
        return [self.age_at_srs,
                self.age_at_first_wish,
                self.age_at_full_time,
                self.duration_of_rlt,
                int(self.femme_child),
                int(self.femme_child_to_others),
                int(self.agp_hundreds)]

# data ... um, let's say "inspired by" Lawrence 2005, "Sexuality Before and
# After Male-to-Female Sex Reassignment Surgery", Table VI

def make_straight():
    return TransWoman(
        attraction_before_srs=Sex.male,
        attraction_after_srs=Sex.male,
        age_at_srs=normal(34, 9.2),
        age_at_first_wish=normal(6.3, 3.4),
        age_at_full_time=normal(28, 8.8),
        duration_of_rlt=normal(63, 63),
        pre_srs_female_partners=normal(0.3, 0.8),
        pre_srs_male_partners=normal(6.6, 8.8),
        femme_child=(random.random() < 0.76),
        femme_child_to_others=(random.random() < 0.76),
        agp_hundreds=(random.random() < 0.18),
        married_before_srs=(random.random() < 0.12),
        sire_before_srs=(random.random() < 0.06)
    )

def make_bi():
    return TransWoman(
        attraction_before_srs=Sex.female,
        attraction_after_srs=Sex.male,
        age_at_srs=normal(45, 8.4),
        age_at_first_wish=normal(9.8, 9.1),
        age_at_full_time=normal(42, 11.3),
        duration_of_rlt=normal(21, 18),
        pre_srs_female_partners=normal(12, 16),
        pre_srs_male_partners=normal(0.7, 1.3),
        femme_child=(random.random() < 0.41),
        femme_child_to_others=(random.random() < 0.21),
        agp_hundreds=(random.random() < 0.52),
        married_before_srs=(random.random() < 0.70),
        sire_before_srs=(random.random() < 0.53)
    )


def make_lesbian():
    return TransWoman(
        attraction_before_srs=Sex.female,
        attraction_after_srs=Sex.female,
        age_at_srs=normal(41, 9.1),
        age_at_first_wish=normal(8, 6),
        age_at_full_time=normal(42, 9.6),
        duration_of_rlt=normal(21, 18),
        pre_srs_female_partners=normal(15, 21),
        pre_srs_male_partners=normal(0.8, 1.8),
        femme_child=(random.random() < 0.45),
        femme_child_to_others=(random.random() < 0.24),
        agp_hundreds=(random.random() < 0.58),
        married_before_srs=(random.random() < 0.74),
        sire_before_srs=(random.random() < 0.42)
    )


def make_data():
    data = []
    target = []
    for group_code, factory in enumerate([make_straight, make_bi, make_lesbian]):
        for _ in range(250):
            data.append(factory().to_data_row())
            target.append(group_code)
    return (numpy.array(data), target)


def plot_data(data, target):
    figure = plot.figure(figsize=(4, 3))
    axes = Axes3D(figure)

    reduced_data = PCA(n_components=3).fit_transform(data)

    axes.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
                 c=target)

    axes.set_xlabel("first eigenvector")
    axes.set_ylabel("second eigenvector")
    axes.set_zlabel("third eigenvector")

    plot.show()


def two_means_prediction(data, target):
    model = KMeans(n_clusters=2)
    model.fit(data)
    prediction_records = [
        # XXX HACK: we don't know which group the KMeans classifier will code
        # as 0 vs. 1, so let's just try both interpretations; the one with a
        # better score will be the correct one
        [model.labels_[i] == target[i] for i in range(len(target))],
        [model.labels_[i] != target[i] for i in range(len(target))]
    ]

    def prediction_fraction(record):
        return len([p for p in record if p])/len(record)

    return max(prediction_fraction(record) for record in prediction_records)


if __name__ == "__main__":
    data, target = make_data()

    # lump together ♀/♀ and ♀/♂ pre-/post- SRS attraction groups together for
    # k-means (k=2) analysis
    bitarget = [1 if t == 2 else t for t in target]
    print("fraction correct: {}".format(two_means_prediction(data, bitarget)))

    plot_data(data, target)
