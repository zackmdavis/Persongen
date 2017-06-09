# Persongen

## exercises in the computational philosophy of gender

### predicting sex from personality

_src/main.rs_

Given _d_ values of sex differences in Big Five personality traits, how easy is it to predict someone's sex from measurements of their personality using a naïve Bayes model?? (_I_ hope it's hard, but the data will decide.)

My friend Sophia points out that I'm doing it wrong: you can't use residuals for sampling; you need to actually use the correlation matrix. But the paper reports that, too (Table 3).

Data source:

* ["Gender Differences in Personality across the Ten Aspects of the Big Five"](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3149680/) by Yanna J. Weisberg, Colin G. DeYoung, and Jacob B. Hirsh

### predicting pre-SRS sexual orientation for trans women

_also/two\_type\_taxonomy.py_

Data source:

* "Sexuality Before and After Male-to-Female Sex Reassignment Surgery" by Anne A. Lawrence
