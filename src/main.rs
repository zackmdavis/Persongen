extern crate rusty_machine;
extern crate rand;

use rand::Rand;
use rand::distributions::Sample;
use rusty_machine::learning::naive_bayes::{self, NaiveBayes};
use rusty_machine::linalg::Matrix;
use rusty_machine::learning::SupModel;
use rusty_machine::stats::dist::gaussian::Gaussian;


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
    agreeableness: f64,
    neuroticism: f64,
}

impl Rand for Person {
    fn rand<R: rand::Rng>(rng: &mut R) -> Self {
        let mut female_agreeableness = Gaussian::from_std_dev(3.89, 0.50);
        let mut male_agreeableness = Gaussian::from_std_dev(3.65, 0.50);
        let mut female_neuroticism = Gaussian::from_std_dev(2.94, 0.67);
        let mut male_neuroticism = Gaussian::from_std_dev(2.68, 0.65);

        let coin: f64 = rng.gen();
        let sex = if coin < 0.5 { Sex::Female } else { Sex::Male };

        match sex {
            Sex::Female => {
                Person {
                    sex: Sex::Female,
                    agreeableness: female_agreeableness.sample(rng),
                    neuroticism: female_neuroticism.sample(rng),
                }
            },
            Sex::Male => {
                Person {
                    sex: Sex::Male,
                    agreeableness: male_agreeableness.sample(rng),
                    neuroticism: male_neuroticism.sample(rng),
                }
            }
        }
    }
}

fn main() {
    let mut randomness = rand::StdRng::new()
        .expect("we should be able to get an RNG");
    let rng = &mut randomness;

    let training_set_size = 100;
    let validation_set_size = 5000;

    let training_set = (0..training_set_size)
        .map(|_| { Person::rand(rng) })
        .collect::<Vec<_>>();

    let validation_set = (0..validation_set_size)
        .map(|_| { Person::rand(rng) })
        .collect::<Vec<_>>();

    let mut measurement_representation = Vec::new();
    let mut classification_representation = Vec::new();
    for training_point in &training_set {
        match training_point.sex {
            Sex::Female => {
                classification_representation.extend_from_slice(&[1., 0.]);
            },
            Sex::Male => {
                classification_representation.extend_from_slice(&[0., 1.]);
            }
        }
        measurement_representation.push(training_point.agreeableness);
        measurement_representation.push(training_point.neuroticism);
    }

    let training_matrix = Matrix::new(training_set_size, 2,
                                      measurement_representation);
    let target_matrix = Matrix::new(training_set_size, 2,
                                    classification_representation);

    let mut model = NaiveBayes::<naive_bayes::Gaussian>::new();
    model.train(&training_matrix, &target_matrix);

    println!("our model:\n{:?}", model);

    let mut validation_representation = Vec::new();
    for validation_point in &validation_set {
        validation_representation.extend_from_slice(
            &[validation_point.agreeableness,
              validation_point.neuroticism]
        );
    }
    let validation_matrix = Matrix::new(validation_set_size, 2,
                                        validation_representation);

    let predictions = model.predict(&validation_matrix);

    let mut hits = 0;
    for (person, prediction) in validation_set.iter()
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

    println!("Predictive accuracy: {}/{} = {}%", hits, validation_set_size,
             (hits as f64)/(validation_set_size as f64) * 100.);
}
