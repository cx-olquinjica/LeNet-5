# LeNet-5

This repository showcases the impact of modern deep learning techniques on the original results reported by the LeNet-5 model, using the MNIST dataset. By exploring techniques such as regularization, optimization, and initialization,I aim to improve the model’s performance and accuracy on this challenging dataset.

## Introduction

LeNet-5 is a classic convolutional neural network (CNN) architecture developed by Yann LeCun et al. It was originally designed for recognizing handwritten digits in the MNIST dataset. 

The primary goal of this repository is to demonstrate how applying modern deep learning techniques can influence the performance of the LeNet-5 model on the MNIST dataset.

## Dataset

The MNIST dataset is widely used for image classification tasks. It comprises 50,000 training images and 10,000 test images, with each image belonging to one of the ten classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

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

 3. Run the training script to train the LeNet-5 model on MNIST:
```
python train.py -c config.json
```

 4. Experiment with different hyperparameters, regularization techniques, optimization algorithms, and weight initialization strategies by modifying the provided configuration files.
 5. Evaluate the trained model on the test set:
```
python test.py
```


## Results

The repository documents the results obtained from different experiments. We compare the original LeNet-5 model’s performance with the modified models that incorporate various deep learning techniques. The impact on accuracy, convergence speed, and generalization ability will be analyzed and presented in the repository.

| Early Stopping (Yes/No) | Activation | Weight Decay| Initialization        | Optimization| # Epoch| Loss    | Accuracy  |  Val Loss| Val Accuracy| 
| ----                    | ---------  | ----        | -----                 | ----        | ----   | ---     | --------- |   -----  | ---------   | 
|Yes = 10                 | Sigmoid    | 0           | Xavier Initialization | Adam        | 72     | 0.00521 | 0.99927   | 0.04306  |  0.98715    |
|No                       |Sigmoid     | 0           | Xavier Initialization | Adam        | 100    | 0.00462 | 0.99951   | 0.04354  |  0.99897    |
|Yes                      |Sigmoid     | 0.001       | Xavier Initialization | Adam        | 63     | 0.12602 | 0.96969   | 0.11491  |  0.97212    |
|Yes                      |ReLU        | 0           | Xavier Initialization | Adam        | 28     | 0.00314 | 0.99898   | 0.04902  |  0.98919    |
|Yes                      |ReLU        | 0.001       | Xavier Initialization | Adam        | 36     | 0.02145 | 0.99361   | 0.03944  |  0.99950    |
|No                       |ReLU        | 0           | Xavier Initialization | Adam        | 100    | 6.02913257830038e-05| 1.0  | 0.05802       | 0.98986|

### Replacing Average Pooling by MaxPooling: 
| Early Stopping | Activation |  Weight Decay | Initialization        | Optimization | # Epochs | Loss  | Accuracy | Val Loss | Val Accuracy  | 
|  -------       |   ---------|     --------- |  --------------       |  ------------|   -------|----   | -------  |  --------|   ------------|
| Yes            |  ReLU      |   0           |  Xavier Initialization|   Adam       |  20      |0.00580|  0.997908| 0.05595  |0.98912        | 

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
