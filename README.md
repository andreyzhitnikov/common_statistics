**This is code repository for the paper: A. Zhitnikov, R. Mulayoff and T. Michaeli, Revealing Common Statistical Behaviors
in Heterogeneous Populations. International Conf. on Machine Learning (ICML 2018)** (http://proceedings.mlr.press/v80/zhitnikov18a.html)

* to install
	* python setup.py install --record files.txt

* to unistall
	* cat files.txt | xargs rm -rf
	
	
**Running examples (tested with python 3.6) :**

* python toy_problem.py --dim 3 --vnum 3000 --visual --show-corrs --b 10

* python common_density_demo.py

* python toy_problem_correlation_coeff.py --visual --num_cores 20

* python plot_single_ours.py 



