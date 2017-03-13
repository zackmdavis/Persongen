import random
from enum import Enum

import attr
from numpy.random import normal

Sex = Enum('Sex', [
    'female', # ♀
    'male', # ♂
])

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
