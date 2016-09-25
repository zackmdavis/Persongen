extern crate rand;
extern crate rusty_machine;

use rand::Rand;
use rand::distributions::Sample;
use rand::distributions::normal::Normal;
use rusty_machine::prelude::*;
use rusty_machine::learning::naive_bayes::{self, NaiveBayes};

// named field pun macro thanks to @durka
// https://github.com/rust-lang/rfcs/pull/1682#issuecomment-241301350
macro_rules! make {
    (@ { $(,)* } $finished:expr) => {
        $finished
    };
    (@ { #[$attr:meta] $($rest:tt)* } $strukt:ident { $($out:tt)* }) => {
        make!(@ { $($rest)* } $strukt { $($out)* #[$attr] })
    };
    (@ { $name:ident: $val:expr, $($rest:tt)* } $strukt:ident { $($out:tt)* }) => {
        make!(@ { $($rest)* } $strukt { $($out)* $name: $val, })
    };
    (@ { $name:ident, $($rest:tt)* } $strukt:ident { $($out:tt)* }) => {
        make!(@ { $($rest)* } $strukt { $($out)* $name: $name, })
    };

    ($strukt:ident { $($field:tt)* }) => {
        make!(@ { $($field)* , } $strukt { })
    }
}


#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Sex {
    Female, // ♀
    Male, // ♂
}

impl Sex {
    fn symbol(&self) -> String {
        match *self {
            Sex::Female => "♀".to_owned(),
            Sex::Male => "♂".to_owned()
        }
    }
}

#[derive(Clone, Debug)]
struct Person {
    sex: Sex,
    enthusiasm: f64,
    assertiveness: f64,
    compassion: f64,
    politeness: f64,
    industriousness: f64,
    orderliness: f64,
    volatility: f64,
    withdrawal: f64,
    intellect: f64,
    openness: f64,
}

struct PopulationPersonalityStructure {
    enthusiasm: Normal,
    residual_assertiveness: Normal,
    compassion: Normal,
    residual_politeness: Normal,
    industriousness: Normal,
    residual_orderliness: Normal,
    volatility: Normal,
    residual_withdrawal: Normal,
    intellect: Normal,
    residual_openness: Normal,
}

impl Rand for Person {
    fn rand<R: rand::Rng>(rng: &mut R) -> Self {
        // Table 2 from http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3149680/
        let female_personality_structure = PopulationPersonalityStructure {
            enthusiasm: Normal::new(3.56, 0.68),
            residual_assertiveness: Normal::new(-0.05, 0.58),
            compassion: Normal::new(4.04, 0.56),
            residual_politeness: Normal::new(0.04, 0.53),
            industriousness: Normal::new(3.21, 0.73),
            residual_orderliness: Normal::new(0.05, 0.57),
            volatility: Normal::new(2.86, 0.77),
            residual_withdrawal: Normal::new(0.05, 0.56),
            intellect: Normal::new(3.48, 0.63),
            residual_openness: Normal::new(0.07, 0.54),
        };

        let male_personality_structure = PopulationPersonalityStructure {
            enthusiasm: Normal::new(3.40, 0.66),
            residual_assertiveness: Normal::new(0.09, 0.57),
            compassion: Normal::new(3.78, 0.60),
            residual_politeness: Normal::new(-0.06, 0.57),
            industriousness: Normal::new(3.25, 0.68),
            residual_orderliness: Normal::new(-0.08, 0.56),
            volatility: Normal::new(2.63, 0.75),
            residual_withdrawal: Normal::new(-0.10, 0.55),
            intellect: Normal::new(3.62, 0.61),
            residual_openness: Normal::new(-0.14, 0.56),
        };

        let coin: f64 = rng.gen();
        let sex = if coin < 0.5 { Sex::Female } else { Sex::Male };
        let mut personality_structure = match sex {
            Sex::Female => female_personality_structure,
            Sex::Male => male_personality_structure,
        };
        let enthusiasm = personality_structure.enthusiasm.sample(rng);
        let assertiveness = enthusiasm +
            personality_structure.residual_assertiveness.sample(rng);
        let compassion = personality_structure.compassion.sample(rng);
        let politeness = compassion +
            personality_structure.residual_politeness.sample(rng);
        let industriousness = personality_structure.industriousness.sample(rng);
        let orderliness = industriousness +
            personality_structure.residual_orderliness.sample(rng);
        let volatility = personality_structure.volatility.sample(rng);
        let withdrawal = volatility +
            personality_structure.residual_withdrawal.sample(rng);
        let intellect = personality_structure.intellect.sample(rng);
        let openness = intellect +
            personality_structure.residual_openness.sample(rng);

        make!(Person { sex,
                       enthusiasm, assertiveness,
                       compassion, politeness,
                       industriousness, orderliness,
                       volatility, withdrawal,
                       intellect, openness })
    }
}

fn main() {
    let mut randomness = rand::StdRng::new()
        .expect("we should be able to get an RNG");
    let rng = &mut randomness;

    let training_set_size = 5000;
    let test_set_size = 5000;

    let training_set = (0..training_set_size)
        .map(|_| { Person::rand(rng) })
        .collect::<Vec<_>>();

    let test_set = (0..test_set_size)
        .map(|_| { Person::rand(rng) })
        .collect::<Vec<_>>();


    let mut training_matrix = Matrix::new(0, 10, Vec::new());
    let mut target_matrix = Matrix::new(0, 2, Vec::new());
    for training_person in &training_set {
        let row = Matrix::new(1, 10, vec![
            training_person.enthusiasm,
            training_person.assertiveness,
            training_person.compassion,
            training_person.politeness,
            training_person.industriousness,
            training_person.orderliness,
            training_person.volatility,
            training_person.withdrawal,
            training_person.intellect,
            training_person.openness,
        ]);
        training_matrix = training_matrix.vcat(&row);
        let target_row = match training_person.sex {
            Sex::Female => Matrix::new(1, 2, vec![1., 0.]),
            Sex::Male => Matrix::new(1, 2, vec![0., 1.])
        };
        target_matrix = target_matrix.vcat(&target_row);
    }

    let mut model = NaiveBayes::<naive_bayes::Gaussian>::new();
    model.train(&training_matrix, &target_matrix)
        .expect("failed to train model");

    let mut test_matrix = Matrix::new(0, 10, Vec::new());
    for test_person in &test_set {
        let row = Matrix::new(1, 10, vec![
            test_person.enthusiasm,
            test_person.assertiveness,
            test_person.compassion,
            test_person.politeness,
            test_person.industriousness,
            test_person.orderliness,
            test_person.volatility,
            test_person.withdrawal,
            test_person.intellect,
            test_person.openness,
        ]);
        test_matrix = test_matrix.vcat(&row);
    }

    let predictions = model.predict(&test_matrix)
        .expect("failed to predict!?");

    let mut hits = 0;
    for (person, prediction) in test_set.iter()
            .zip(predictions.iter_rows()) {
        let predicted_sex = person.sex;
        let actual_sex = if prediction[0] == 1. {
            Sex::Female
        } else {
            Sex::Male
        };
        let accurate = predicted_sex == actual_sex;
        if accurate {
            hits += 1;
        }
        println!("Predicted: {}; Actual: {}; Accurate? {:?}", predicted_sex.symbol(), actual_sex.symbol(), accurate);
    }

    println!("Predictive accuracy: {}/{} = {}%", hits, test_set_size,
             (hits as f64)/(test_set_size as f64) * 100.);
}
