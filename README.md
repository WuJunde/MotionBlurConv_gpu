## Integrating Neural Networks Into the Blind Deblurring Framework to Compete With the End-to-End Learning-Based Methods

arxiv: https://arxiv.org/abs/1903.02731

accepted by IEEE Transactions on Image Processing

Here is the GPU version deconvolution code of the paper

### Main Idea

This paper propose a method to embed the deep neural network to the traditional debluring framework. 

We propose SEN to estimate the blur kernel and RP-GAN to recover the clear images.

An illustration figure:

### ![system](https://github.com/WuJunde/MotionBlurConv_gpu/blob/master/system.png)Results

Go-pro dataset:

![gopro_car](https://github.com/WuJunde/MotionBlurConv_gpu/blob/master/gopro_car.png)

Images from top to bottom are blurre dimage, results of Kupynet al. [1], Nahet al. [2], Taoet al. [3] and ours, respectively.



Kohler dataset:

![kohler](https://github.com/WuJunde/MotionBlurConv_gpu/blob/master/kohler.png)

Images  from  left  to  right  are  blurry  image,  results  of  Kupynet al.  [1],  Nahet al.  [2],  Taoet al.  [3]  and  ours, respectively

Real blur photo:

![real](https://github.com/WuJunde/MotionBlurConv_gpu/blob/master/real.png)

Images from top to bottom are the blurry image, result of Kupynet al. [1], Nahet al. [2], Taoet al. [3] and ours, respectively.



[1]  O.  Kupyn,  V.  Budzan,  M.  Mykhailych,  D.  Mishkin,  and  J.  Matas.  De-blurgan: Blind motion deblurring using conditional adversarial networks.2017.

[2]  S. Nah, T. H. Kim, and K. M. Lee. Deep multi-scale convolutional neuralnetwork for dynamic scene deblurring.  pages 257–265, 2016.

[3]  X.  Tao,  H.  Gao,  X.  Shen,  J.  Wang,  and  J.  Jia.  Scale-recurrent  networkfor deep image deblurring.  InIEEE Conference on Computer Vision andPattern Recognition (CVPR), 2018.
