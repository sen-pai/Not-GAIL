# Not-GAIL
To install each library here use ``pip install -e``

Make sure you are not using python 3.9, I faced many installation problems with it. Python 3.7 or 3.8 is better.

#### New files added for Anything
``libraries\imitation\src\imitation\algorithms\anything_module.py``\
``libraries\imitation\src\imitation\rewards\neg_discrim_nets.py``

#### New files added for Something
``libraries\imitation\src\imitation\algorithms\something_module.py``\
``libraries\imitation\src\imitation\utils\reward_wrapper.py``


#### Test Script
To run the script first install both imitation and sb3.
Then you need to re-install imiation from ``libraries\imiation``
This is because it needs to register the new files that are added.

run ``python test_anything_module.py``

``cartpole_proper.pkl`` are trajectories from a trained PPO that solved Cartpole-v1 env. 

Ideally you should see the ``gen`` log with increasing reward and ``neg_gen`` log with reducing reward. 