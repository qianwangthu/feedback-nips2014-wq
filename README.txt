***************************************************************
Code for  Attentional Neural Network: Feature Selection Using Cognitive Feedback 

For the latest information, please contact us:
Qian Wang <qianwang.thu@gmail.com>
Jiaxing Zhang <jiaxz@microsoft.com>
Sen Song <sen.song@gmail.com>
Zheng Zhang <zz@nyu.edu>,<zz17@nyu.edu>
===============================================================

1.data files: We removed data files because of the limitation of attachment size, but you can generate them yourself. 'data' folder contains two files:
	mnist_uint8.mat: contains 4 variables named with test_x, test_y, train_x, train_y. They are all from the MNIST digit recognition dataset (http://yann.lecun.com/exdb/mnist/);
		- test_x:  10000*784 uint8, image data
		- test_y:  10000*10  uint8, label data
		- train_x: 60000*784 uint8, image data
		- train_y: 60000*10  uint8, label data
	background_image.mat: contains 1 variable named with T
		- T: 70000*784 uint8, random patches cropped from natural images. You can generate it yourself from any natural image. We used images from MNIST Variations dataset (http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/MnistVariations).
		
		
2.folders:
	- util:		   some common function. You need to add it to MATLAB path before run any other code.
	- data:		   containing data files for digits and background. 
	- model:	   containing trained model. Here we release a well-trained model in 'feedback_hf_p5_model.mat'as a sample, which is trained in mnist-background-image.
	- pretrain:	   pretraining by RBM. Begin with 'main_sparserbm.m'.
	- train-feedback:  training feedback weights. Begin with 'main_feedback.m'.
	- test-background: classification in MNIST Variations dataset (digits with background).
	- test-mnist2:	   classification in MNIST2 dataset.
	
	