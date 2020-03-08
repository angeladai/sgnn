# SG-NN

SG-NN presents a self-supervised approach that converts partial and noisy RGB-D scans into high-quality 3D scene reconstructions by inferring unobserved scene geometry. For more details please see our paper [
SG-NN: Sparse Generative Neural Networks for Self-Supervised Scene Completion of RGB-D Scans](https://arxiv.org/pdf/1912.00036.pdf).

[<img src="sgnn.jpg">](https://arxiv.org/abs/1912.00036)


## Code
### Installation:  
Training is implemented with [PyTorch](https://pytorch.org/). This code was developed under PyTorch 1.1.0.

### Training:  
* See `python train.py --help` for all train options. 
* Trained models: Coming soon!

### Testing
* See `python test_scene.py --help` for all test options. 


### Data:
Train and test data to come soon.


## Citation:  
If you find our work useful in your research, please consider citing:
```
@article{dai2019sgnn,
 title={SG-NN: Sparse Generative Neural Networks for Self-Supervised Scene Completion of RGB-D Scans},
 author = {Dai, Angela and Diller, Christian and Nie{\ss}ner, Matthias},
 journal = {arXiv preprint arXiv:1912.00036},
 year = {2019}
}
```
