# assignment2_GAN
# learning probability density function using gan

## objective
the objective of this assignment is to learn an unknown probability density function
of a transformed random variable using only data samples and a generative adversarial
network (gan). no analytical form of the pdf is assumed.

the no2 concentration values are taken from an air quality dataset and transformed
using a given non-linear transformation. a gan is then trained to learn the
distribution of the transformed variable.


## dataset
- dataset: india air quality data
- feature used: no2 concentration
- missing values are removed before processing

## transformation of data
each no2 value `x` is transformed into a new variable `z` using the given equation:


where:
- university roll number (r) = 102303902  
- a_r = 0.5 × (r mod 7) = 0.5  
- b_r = 0.3 × ((r mod 5) + 1) = 0.9  

this transformation makes the distribution of `z` non-linear and analytically
unknown, which motivates the use of gan-based density learning.


## gan methodology

### generator
- input: random noise sampled from normal distribution n(0,1)
- architecture:
  - linear layer (1 → 16)
  - relu activation
  - linear layer (16 → 1)
- output: fake samples of transformed variable `z`

### discriminator
- input: real or fake `z` values
- architecture:
  - linear layer (1 → 16)
  - leaky relu activation
  - linear layer (16 → 1)
  - sigmoid activation
- output: probability of sample being real

the generator and discriminator are trained using binary cross entropy loss in an
adversarial manner.


## training details
- training is done using only samples of transformed variable `z`
- no parametric pdf (gaussian, exponential, etc.) is assumed
- optimizer: adam
- number of epochs: 1500
- batch size: 256

during training, the discriminator learns to distinguish real and fake samples,
while the generator learns to fool the discriminator.


## pdf approximation
after training the gan:
- a large number of samples are generated using the trained generator
- a histogram-based density estimation is used to approximate the probability
  density function of `z`

this histogram represents the learned pdf of the transformed variable.


## results and observations

### mode coverage
the generator is able to capture the main modes of the transformed data distribution.
however, extreme tail values are slightly compressed due to the simplicity of the
network architecture.

### training stability
initial training shows fluctuations between generator and discriminator losses,
which is expected in gan training. after sufficient epochs, training becomes more
stable.

### quality of generated distribution
the generated samples follow a similar shape and scale as the real transformed data.
the histogram-based pdf closely matches the empirical distribution obtained from the
dataset.


## conclusion
this assignment demonstrates that a gan can successfully learn an unknown probability
density function using only data samples. the approach avoids any parametric
assumptions and relies purely on adversarial learning to model the distribution of the
transformed variable.


## files included
- colab notebook (.ipynb) containing complete implementation
- generated pdf plot (histogram)
- this readme file explaining methodology and results
