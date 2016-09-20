#![allow(unused_imports)]

extern crate rusty_machine;
extern crate rand;

use rand::Rand;
use rand::distributions::Sample;
use rusty_machine::learning::naive_bayes::{self, NaiveBayes};
use rusty_machine::linalg::Matrix;
use rusty_machine::learning::SupModel;
use rusty_machine::stats::dist::gaussian::Gaussian;


#[derive(Debug, Eq, PartialEq)]
enum Sex {
    Female, // ♀
    Male, // ♂
}

#[derive(Debug)]
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
    let mut rng = rand::StdRng::new().expect("we should be able to get an RNG");
    let person = Person::rand(&mut rng);
    println!("{:?}", person);
}
