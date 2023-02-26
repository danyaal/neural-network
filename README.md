# Neural Network
A simple Python script that implements a neural network

I've also implemented two use cases for this neural network:
- Handwritten number recognition
- Handwritten alphabet recognition
- Facial expression recognition

## Number recognition
`% python number_recognition.py`

This neural network is trained with the [MNIST database](http://yann.lecun.com/exdb/mnist/) and can be tested with a provided data set or with your own uploaded image! This neural network has adjustable parameters that can be tweaked to around 98% accuracy at recognizing numbers (and definitely could be optimized further)!

![Example](util/assets/5.png)

The neural network would recognize this image as the number `5`

## Alphabet recognition
`% python alphabet_recognition.py`

This neural network is trained with the [NIST database](https://www.nist.gov/srd/nist-special-database-19) and can be tested with your own uploaded image! This neural network has adjustable parameters that can be tweaked.

![Example](util/assets/p.png)

The neural network would recognize this image as a `P`

## Facial expression recognition
`% python smile_recognition.py`

This neural network is trained with my own custom made data set and can be tested with your own uploaded image! The facial recognition isn't always the most accurate because my training data is pretty limited, but it's still cool!

![Example](util/assets/smile1.png)

The neural network would recognize this image as a `smile`

## Usage
If you want to run this yourself, you'll want to unzip `alphabet_assets.zip` and `numbers_assets.zip` before running the alphabet or number recognition code. Those files were too large for Github, so I had to compress them for 🙃.

## Learn more
Check out [https://en.wikipedia.org/wiki/Artificial_neural_network](https://en.wikipedia.org/wiki/Artificial_neural_network) to learn more about them.

## Acknowledgments
Shout out to Tariq Rashid and his great [book](https://www.amazon.com/Make-Your-Own-Neural-Network/dp/1530826608/r) for introducing me to neural networks!!
