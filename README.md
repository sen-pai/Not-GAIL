# Not-GAIL
To install each library here use ``pip install -e``

Make sure you are not using python 3.9, I faced many installation problems with it. Python 3.7 or 3.8 is better.

#### New files added for Anything
``libraries\imitation\src\imitation\algorithms\anything_module.py``\
``libraries\imitation\src\imitation\rewards\neg_discrim_nets.py``


#### Test Script
To run the script first install both imitation and sb3.
Then you need to re-install imiation from ``libraries\imiation``
This is because I have added 2 new files for anything.

run ``python test_anything_module.py``

``cartpole_proper.pkl`` are trajectories from a trained PPO that solved Cartpole-v1 env. 

Ideally you should see the ``gen`` log with increasing reward and ``neg_gen`` log with reducing reward. 