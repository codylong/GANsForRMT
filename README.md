# GANsForRMT

The motivation for this project is to understand whether machine learning, in particular 
generative models, can help to derive useful random matrix approximations to string theory data. 
The two works critical to this project are:
- [Wasserstein GAN](https://arxiv.org/abs/1701.07875), which is the GAN architecture that was most sucessful.
- [Heavy Tails](https://arxiv.org/pdf/1407.0709.pdf), which lays out the framework for the data of interest.
The final results are published, and can be found at [Statistical Predictions in String Theory and Deep Generative Models](https://inspirehep.net/literature/1773864).

The data of interest is obtained via triangulations of 4d reflexive polytopes in the 
[Kreuzer-Skarke database](http://hep.itp.tuwien.ac.at/~kreuzer/CY/), as well as
a python-to-geometry pipeline that can be found in Cody Long's github. 
