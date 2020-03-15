## ImageClassification_RadialBasisFunctions_SelfOrganizingMaps
Image classification on MNIST Hand-written Digits Dataset and function aproximation (regression) via full custom Radial Basis Functions (Exact Interpolation + Fixed Centers Selected at Random Method + Regularization) and Self Organizing Maps implementation.

__For usage instructions please check [Usage README](https://github.com/SamyuelDanyo/ImageClassification_RadialBasisFunctions_SelfOrganizingMaps/blob/master/docs/README.txt)__

__For full documentation - system design, experiments & findings please read [ImageClassification_RadialBasisFunctions_SelfOrganizingMapsDoc Report](https://github.com/SamyuelDanyo/ImageClassification_RadialBasisFunctions_SelfOrganizingMaps/blob/master/docs/ImageClassification_RadialBasisFunctions_SelfOrganizingMapsDoc.pdf)__

__[GutHub Pages Format](https://samyueldanyo.github.io/image-classification-radial-basis-functions-self-organizing-maps/)__

## Introduction
Few experiments are undertaken:

__Q1. Function Approximation with RBFN__

  __*𝑦 = 1.2sin(𝜋𝑥)−cos(2.4𝜋𝑥), 𝑓𝑜𝑟 𝑥 ∈ [−1.6,1.6]*__
  
  (a) Exact interpolation method (Gaussian, standard deviation of 0.1).
  
  (b) Fixed Centers Selected at Random
  
  (c) Regularization study
  
__Q2. Handwritten Digits Classification using RBFN__

  (a) Exact Interpolation Method + regularization (Gaussian function, standard deviation of 100)
  
  (b) Gaussian function with standard deviation of 100 + varying width
  
  (c) Xlassical “K-Mean Clustering" with 2 centers.
  
__Q3. Self-Organizing Map (SOM)__

  (a) Map a 1-dimensional output layer of 40 neurons to a “hat” (sinc function) + visualisation.
  
  (b) Maps a 2-dimensional output layer of 64 (i.e. 8×8) neurons to a “circle” + visualisation.
  
  (c) Cluster and classifiy handwritten digits.
  
    + Visualisation of conceptual/semantic map of the trained SOM and the trained weights of each output neuron on a 10×10 map.
    
## Figures:
### Radial Basis Functions

![Radial Basis Function SIN Aproximation](/res/rbf_gi_regr.png) ![Radial Basis Function SIN Aproximation](/res/rbf_fcsr_regr.png)

![Radial Basis Function SIN Aproximation](/res/rbf_fcsr_reg_regr.png) ![Radial Basis Function Architecture](/res/rbf_arch_mnist.png)

### Self Organizing Maps

![Self Organizing Maps SIN Aproximation](/res/som_sin_1.png) ![Self Organizing Maps SIN Aproximation](/res/som_sin_2.png)
![Self Organizing Maps SIN Aproximation](/res/som_sin_3.png) ![Self Organizing Maps SIN Aproximation](/res/som_sin_4.png)
![Self Organizing Maps SIN Aproximation](/res/som_sin_5.png) ![Self Organizing Maps SIN Aproximation](/res/som_sin_6.png)
![Self Organizing Maps SIN Aproximation](/res/som_sin_7.png) ![Self Organizing Maps SIN Aproximation](/res/som_sin_8.png)


![Self Organizing Maps CIRCLE Aproximation](/res/som_circl_1.png) ![Self Organizing Maps CIRCLE Aproximation](/res/som_circl_2.png)
![Self Organizing Maps CIRCLE Aproximation](/res/som_circl_3.png) ![Self Organizing Maps CIRCLE Aproximation](/res/som_circl_4.png)
![Self Organizing Maps CIRCLE Aproximation](/res/som_circl_5.png) ![Self Organizing Maps CIRCLE Aproximation](/res/som_circl_6.png)
![Self Organizing Maps CIRCLE Aproximation](/res/som_circl_7.png) ![Self Organizing Maps CIRCLE Aproximation](/res/som_circl_8.png)

![Self Organizing Maps MNIST Classification](/res/SOM_label_plot.png) ![Self Organizing Maps MNIST Classification](/res/SOM_image_plot.png)

![Self Organizing Maps MNIST Classification](/res/SOM_weights_plot.png) ![Self Organizing Maps MNIST Classification](/res/som_train.png)
