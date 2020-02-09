# AI learning and training articles and codes

The goal of this reposistory is to summarize the AI technologies and implementations, please refer to [outline](docs/outline.md) for the overall summaries of the AI topics.

## The components of the repository

### data component


### documents and samples
  The documents for the AI articles are mainly written using markdown format, for easily demostrating the examples, we use docker to setup pytorch, jupyter notebook and other libraries environments. please refer to [INSTASLL](./docs/deep_learning/pytorch/manual.md) for the preparation of the environments setup.

#### Math
- [Linear Algebra Review and Reference(1)](http://note.youdao.com/noteshare?id=b7a6cfe77e3906bdb5639d1acec3c88c)
- [Linear Algebra Review and Reference(2)](http://note.youdao.com/noteshare?id=a3dda151febf0da4dc17df5ec918b41b)
- [Entropy_Cross_Entropy_KL_Divergence](http://note.youdao.com/noteshare?id=b996997b7918d6c3fb9f6aa6813aa675)
- [Taylor_expansion_Multi_Variables_Functions_extremum](http://note.youdao.com/noteshare?id=951f44d73e0777672abffc7ef891f2ea)

#### Machine Learning
  - **Part I** [Linear_Regression](http://101.132.45.94/2020/01/30/linear-regression/)
  - **Part II** Classification and logistic regression
      - [Logistic_Regression](http://note.youdao.com/noteshare?id=a62bb63c6a049ce5e0cdc8abfe8ba3fd)
      - Newton's Method
  - **Part III** Generalized Linear Models
  - **Part IV** Generative Learning algorithms
      - Gaussian discriminant analysis
      - Naive Bayes
  - **Part V** Kernel Methods
  - **Part VI** SVM
  - **Part VI** Learning Theory
      - Bias/variance tradeoff
      - Preliminaries
      - The case of finite H
      - The case of infinite H
      - Regularization and model selection
  - **Adaboost**
  - **Decision Tree**
  - **Random Forest**
  - **Tree Boosting**
      - GBDT
      - XGBoost
      - LightGBM
  - **HMM**
  - **MRF**
- The k-means clustering algorithm
- Mixtures of Gaussians and the EM algorithm
- **Part IX** The EM algorithm
- **Part X** Factor analysis
- **Part XI** Principal components analysis
- **Part XII** Independent Components Analysis
- **Part XIII** Reinforcement Learning and Control

#### Deep Learning
- CNN
- RNN
    - [LSTM]
        - [公式及实现](http://note.youdao.com/noteshare?id=84b5e5bad8db62a45682c5b928a4e9a8&sub=5708D04E282940B3922FAA10C096CBE8)
- GAN
- VAE
- Understanding Deep Learning

#### Computer Vision
- Object Detection
- Semantic/Instance Segmentation
- Image to image translation
    - [github](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) 
    - [pix2pix model definition](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/pix2pix_model.py)

## references
- [stanford course](http://cs229.stanford.edu/syllabus.html)
- [ufldl](http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial)/(http://ufldl.stanford.edu/tutorial/)
- [夕小瑶](https://www.jiqizhixin.com/users/22850d06-ec08-47b6-87c6-9d065a38c83c)
- [latex常用公式](https://blog.csdn.net/oBrightLamp/article/details/83964331)
- [deep-learning-glossary](http://www.wildml.com/deep-learning-glossary/)
- [深度学习算法与编程](https://blog.csdn.net/oBrightLamp/article/details/85067981)
- [AndrewNg 机器学习笔记](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/tree/master/markdown)


## TODO items
- [ ] Windows docker environment setup
- [ ] Linear Regression sample code(adding an example from blog's link besides autograd.ipynb document)
