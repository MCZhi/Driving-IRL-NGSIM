# Driving-IRL-NGSIM
This repo is the implementation of the paper "Driving Behavior Modeling using Naturalistic Human Driving Data with Inverse Reinforcement Learning". It contains NGSIM env that can replay vehicle trajectories in the NGSIM dataset while also simulate some interactive behaviors, as well as inverse reinforcement learning (IRL) implementation in this paper for learning driver's reward function.

**Driving Behavior Modeling using Naturalistic Human Driving Data with Inverse Reinforcement Learning**
<br> [Zhiyu Huang](https://mczhi.github.io/), [Jingda Wu](https://scholar.google.com/citations?user=icu-ZFAAAAAJ&hl=en), [Chen Lv](https://scholar.google.com/citations?user=UKVs2CEAAAAJ&hl=en) 
<br> [AutoMan Research Lab, Nanyang Technological University](https://lvchen.wixsite.com/automan)
<br> **[[Paper]](https://ieeexplore.ieee.org/document/9460807)**&nbsp;**[[arXiv]](https://arxiv.org/abs/2010.03118)**

## Getting started
Install the dependent package
```shell
pip install -r requirements.txt
```

Download the NGSIM dataset from [this website](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj) (export to csv file) and run dump_data.py along with the path to the downloaded csv file (may take a while)
```shell
python dump_data.py [YOUR PATH]/Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data.csv
```

Run IRL personalized or IRL general
```shell
python personal_IRL.py 
```
```shell
python general_IRL.py 
```

## Reference
If you find this repo to be useful in your research, please consider citing our work
```
@article{huang2021driving,
  title={Driving Behavior Modeling Using Naturalistic Human Driving Data With Inverse Reinforcement Learning},
  author={Huang, Zhiyu and Wu, Jingda and Lv, Chen},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2021},
  publisher={IEEE}
}
```

## License
This repo is released under the MIT License. The NGSIM data processing code is borrowed from [NGSIM interface](https://github.com/Lemma1/NGSIM-interface). The NGSIM env is built on top of [highway env](https://github.com/eleurent/highway-env) which is released under the MIT license.
