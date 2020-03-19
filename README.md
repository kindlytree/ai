# AI related articles and samples

The goal of this reposistory is to summarize the AI technologies and implementations.

## The components of the repository

### Datasets


### Articles and samples
  The documents for the AI articles are mainly written using markdown format, for easily demostrating the examples, we use docker to setup pytorch, jupyter notebook and other libraries environments. please refer to [INSTASLL](./docs/deep_learning/pytorch/manual.md) for the preparation of the environments setup.

#### Math
- [Linear Algebra Review and Reference(1)](http://note.youdao.com/noteshare?id=b7a6cfe77e3906bdb5639d1acec3c88c)
- [Linear Algebra Review and Reference(2)](http://note.youdao.com/noteshare?id=a3dda151febf0da4dc17df5ec918b41b)
- [Entropy_Cross_Entropy_KL_Divergence](http://note.youdao.com/noteshare?id=b996997b7918d6c3fb9f6aa6813aa675)
- [Taylor_expansion_Multi_Variables_Functions_extremum](http://note.youdao.com/noteshare?id=951f44d73e0777672abffc7ef891f2ea)

#### Machine Learning
- Supervised Learning
  - **Part I** [Linear_Regression](http://101.132.45.94/2020/01/30/linear-regression/) [code](https://github.com/kindlytree/ai/blob/master/samples/ml/linear_regression.ipynb)
  - **Part II** Classification and logistic regression
      - [Logistic_Regression](http://note.youdao.com/noteshare?id=a62bb63c6a049ce5e0cdc8abfe8ba3fd) [code](https://github.com/kindlytree/ai/blob/master/samples/ml/logistic_regression.ipynb)
      - [Newton's Method](http://note.youdao.com/noteshare?id=57e9b323d4ae19c215c421fcac32b638) [code](https://github.com/kindlytree/ai/blob/master/samples/ml/newton_method.ipynb)
  - **Part III** [Generalized Linear Models](http://note.youdao.com/noteshare?id=b814a849cf4752746518d4f63ef0d79c) [softmax regression code](https://github.com/kindlytree/ai/blob/master/samples/ml/softmax_regression.ipynb)
  - **Part IV** [Generative Learning algorithms](http://note.youdao.com/noteshare?id=179205e43731362a960bf52236599fa9)
      - [Gaussian discriminant analysis](http://note.youdao.com/noteshare?id=7a34e72665581d2d379ac9a9cdebd0ce)
      - [Naive Bayes](http://note.youdao.com/noteshare?id=0ca8c256d4dcb349dd32b155594426ea)
  - **Part V** [Kernel Methods](http://note.youdao.com/noteshare?id=5de8fb8eaa20e53517671b7d706bd6c6)
  - **Part VI** [SVM](http://note.youdao.com/noteshare?id=04eb156cc9eb0137844a2a381f3f1668)
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
          - [CART](http://note.youdao.com/noteshare?id=922bd61daea279fed55ac3359c4f9cd3)
      - XGBoost
      - LightGBM
  - **HMM**
  - **MRF**
  - **Neural Network**
      - [Back Propagation](http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/)
- The k-means clustering algorithm
- Mixtures of Gaussians and the EM algorithm
    - [GMM](http://note.youdao.com/noteshare?id=611be89d2eeb9c40c79bc5f5e86bc022)
- **Part IX** The EM algorithm
- **Part X** Factor analysis
- **Part XI** Principal components analysis
- **Part XII** Independent Components Analysis
- **Part XIII** Reinforcement Learning and Control

#### Deep Learning
- CNN
- RNN
    - LSTM
        - [公式及实现](http://note.youdao.com/noteshare?id=84b5e5bad8db62a45682c5b928a4e9a8&sub=5708D04E282940B3922FAA10C096CBE8)
- GAN
- VAE

#### Computer Vision
- Object Detection
- Semantic/Instance Segmentation
- Image to image translation
    - [github](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) 
    - [pix2pix model definition](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/pix2pix_model.py)

#### AI practices
- [Data realted issus](http://www.kindlytree.com/2020/02/25/data-related-problems-for-ai/)

#### pytorch samples
- (强烈推荐)[pytorch入门与实践](https://github.com/chenyuntc/pytorch-book)
- [PyTorchDocs](https://github.com/fendouai/PyTorchDocs)
- Our samples 
  - location: ./samples/pytorch
  - how to use
    - docker environment setup [Manual](./docs/deep_learning/pytorch/manual.md)  
    - open samples on local web browser(http://localhost:1234) 
### Papers and related code
- [cvpr](https://github.com/Sophia-11/Awesome-CVPR-Paper)

## references
- [stanford course](http://cs229.stanford.edu/syllabus.html)
- [ufldl](http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial)/(http://ufldl.stanford.edu/tutorial/)
- [AndrewNg 机器学习笔记](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/tree/master/markdown)
- [微软AI课程](https://github.com/microsoft/ai-edu)


## TODO items
- [ ] css229 articles and samples preparation

