# LeNet-5

This repository showcases the impact of modern deep learning techniques on the original results reported by the LeNet-5 model, using the CIFAR-10 dataset. By exploring techniques such as regularization, optimization, and initialization,I aim to improve the model’s performance and accuracy on this challenging dataset.

## Introduction

LeNet-5 is a classic convolutional neural network (CNN) architecture developed by Yann LeCun et al. It was originally designed for recognizing handwritten digits in the MNIST dataset. In this project,I adapt the LeNet-5 model to work with the CIFAR-10 dataset, which consists of 60,000 32x32 color images across 10 classes.

The primary goal of this repository is to demonstrate how applying modern deep learning techniques can influence the performance of the LeNet-5 model on the more complex CIFAR-10 dataset.

## Dataset

The CIFAR-10 dataset is widely used for image classification tasks. It comprises 50,000 training images and 10,000 test images, with each image belonging to one of the ten classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## Model Architecture

The LeNet-5 architecture consists of seven layers, including two convolutional layers, two average pooling layers, and three fully connected layers. The model architecture can be summarized as follows:

 1. Convolutional Layer (6 filters, kernel size 5x5, stride 1)
 2. Average Pooling Layer (pool size 2x2, stride 2)
 3. Convolutional Layer (16 filters, kernel size 5x5, stride 1)
 4. Average Pooling Layer (pool size 2x2, stride 2)
 5. Fully Connected Layer (120 neurons)
 6. Fully Connected Layer (84 neurons)
 7. Output Layer (10 neurons)

## Techniques Explored

This repository explores the impact of several modern deep learning techniques on the LeNet-5 model’s original results:

 1. Regularization: Utilizing regularization techniques such as L1 or L2 regularization to prevent overfitting and improve generalization performance.
 2. Optimization: Applying advanced optimization algorithms like Adam, RMSprop, or SGD with momentum to enhance the training process and convergence speed.
 3. Initialization: Exploring different weight initialization strategies such as Xavier or He initialization to improve the model’s learning ability.

## Usage

To replicate and experiment with the results, follow the steps outlined below:

 1. Clone this repository:

```
git clone git@github.com:cx-olquinjica/LeNet-5.git 
```

 2. Install the necessary dependencies. It is recommended to use a virtual environment:
```
cd LeNet-5 
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

 3. Run the training script to train the LeNet-5 model on CIFAR-10:
```
python train.py
```

 4. Experiment with different hyperparameters, regularization techniques, optimization algorithms, and weight initialization strategies by modifying the provided configuration files.
 5. Evaluate the trained model on the test set:
```
python test.py
```


## Results

The repository documents the results obtained from different experiments. We compare the original LeNet-5 model’s performance with the modified models that incorporate various deep learning techniques. The impact on accuracy, convergence speed, and generalization ability will be analyzed and presented in the repository.

| Model | Activation | Regularization| Initialization| Optimization| # Epoch| Loss| Accuracy|  Val Loss| Val Accuracy| 
| ----             | --------- | ---- | ----- | ---- | ---- | ---     | --------- |   ----- | --------- | 
|LeNet-5 | ?         | RLHF | 92.0  | 42.5 | 86.4 | -       | 67.0      |   68.7* | 43.4      |
| claude-v1.3      | ?         | RLHF | 81.8* | -    | 74.8*| 67.3*   | -         |   54.2* | 24.9      |
| PaLM-2           | ?         | Base | 80.7  | 34.3 | 78.3 | 78.1    | -         |   -     | 31.8      |
| gpt-3.5-turbo    | ?         | RLHF | 74.9* | -    | 67.3*| 70.1*   | 48.1      |   54.4* | 30.2      |
| claude-instant   | ?         | RLHF | 70.8* | -    | -    | 66.9*   | -         |   45.9* | 23.6      |
| text-davinci-003 | ?         | RLHF | -     | -    | 64.6 | 70.7    | -         |   -     | 22.8      |
| code-davinci-002 | ?         | Base | 66.6  | 19.1 | 64.5 | 73.7    | 47.0      |   -     | -         |
| text-davinci-002 | ?         | SIFT | 55.4  | -    | 60.0 | 67.2    | -         |   -     | 16.6      |
| Minerva          | 540B      | SIFT | 58.8  | 33.6 | -    | -       | -         |   -     | -         |
| Flan-PaLM        | 540B      | SIFT | -     | -    | 70.9 | 66.3    | -         |   -     | -         |
| Flan-U-PaLM      | 540B      | SIFT | -     | -    | 69.8 | 64.9    | -         |   -     | -         |
| PaLM             | 540B      | Base | 56.9  | 8.8  | 62.9 | 62.0    | 26.2      |   -     | -         |
| LLaMA            | 65B       | Base | 50.9  | 10.6 | 63.4 | -       | 23.7      |   38.8* | -         |
| PaLM             | 64B       | Base | 52.4  | 4.4  | 49.0 | 42.3    | -         |   -     | -         |
| LLaMA            | 33B       | Base | 35.6  | 7.1  | 57.8 | -       | 21.7      |   -     | -         |
| InstructCodeT5+  | 16B       | SIFT | -     | -    | -    | -       | 35.0      |   -     | 11.6      |
| StarCoder        | 15B       | Base | 8.4   | 15.1 | 33.9 | -       | 33.6      |   -     | 12.2      |
| Vicuna           | 13B       | SIFT | -     | -    | -    | -       | -         |   -     | 12.9      |
| LLaMA            | 13B       | Base | 17.8  | 3.9  | 46.9 | -       | 15.8      |   -     | -         |
| Flan-T5          | 11B       | SIFT | 16.1* | -    | 48.6 | 41.4    | -         |   -     | -         |
| Alpaca           | 7B        | SIFT | -     | -    | -    | -       | -         |   -     | 13.5      |
| LLaMA            | 7B        | Base | 11.0  | 2.9  | 35.1 | -       | 10.5      |   -     | -         |
| Flan-T5          | 3B        | SIFT | 13.5* | -    | 45.5 | 35.2    |    -      |   -     | -         |
## Documentation

The repository includes detailed documentation on the experiments conducted and their corresponding results. Each experiment is clearly described, including the specific technique applied (regularization, optimization, or initialization), the hyperparameters used, and any modifications made to the LeNet-5 architecture.

The documentation also includes visualizations such as accuracy and loss plots, comparisons of different techniques, and explanations of the observed improvements or trade-offs.

## Contributions

Contributions to this repository are welcome. If you have any suggestions, bug fixes, or additional techniques that can enhance the LeNet-5 model’s performance on CIFAR-10, please submit a pull request. Your contributions will help improve the understanding and application of modern deep learning techniques.

## License

This project is licensed under the MIT License. See the [LICENSE](fhhfoh) file for more information.

## Acknowledgments

We would like to acknowledge the original authors of the LeNet-5 model and the creators of the CIFAR-10 dataset. Their contributions have significantly advanced the field of deep learning and image classification.

## Conclusion

This repository serves as a comprehensive showcase of how modern deep learning techniques, including regularization, optimization, and initialization, impact the performance of the LeNet-5 model on the CIFAR-10 dataset. By examining and comparing the results of various experiments, we gain insights into the effectiveness of different techniques and their influence on accuracy and generalization.

We hope that this repository inspires further exploration and experimentation with deep learning techniques to improve the performance of CNN models on complex image classification tasks.
