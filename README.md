# Not-GAIL
To install each library here use ``pip install -e``

Make sure you are not using python 3.9, I faced many installation problems with it. Python 3.7 or 3.8 is better.

#### New files added for Anything
``libraries\imitation\src\imitation\algorithms\anything_module.py``\
``libraries\imitation\src\imitation\rewards\neg_discrim_nets.py``

#### New files added for Something
``libraries\imitation\src\imitation\algorithms\something_module.py``\
``libraries\imitation\src\imitation\utils\reward_wrapper.py``

#### New Functions added in BC 
Find them in ``libraries\imitation\src\imitation\algorithms\bc.py``
``_calculate_only_l2_loss``: Plain L2 loss.
``_calculate_crossentropy_loss``: One hot actions and return CE loss.


#### Minigrid Tips
If using an image as obs, wrap the env as follows:
``env = gym.make('MiniGrid-Empty-Random-6x6-v0')``
``env = wrappers.RGBImgObsWrapper(env)``
``env = wrappers.ImgObsWrapper(env)``

Or for a venv, use 
``from imitation.util import util`` 
``venv = util.make_vec_env('MiniGrid-Empty-Random-6x6-v0', n_envs=1, post_wrappers= [wrappers.RGBImgObsWrapper, wrappers.ImgObsWrapper])``


Similarly, for a flat obs use: 
``env = gym.make('MiniGrid-Empty-Random-6x6-v0')``
``env = wrappers.FlatObsWrapper(env)``


Pass ``FlatObsWrapper`` in post_wrappers for a venv
#### Test Script
To run the script first install both imitation and sb3.
Then you need to re-install imitation from ``libraries\imiation``
This is because it needs to register the new files that are added.

run ``python test_anything_module.py``

``cartpole_proper.pkl`` are trajectories from a trained PPO that solved Cartpole-v1 env. 

Ideally you should see the ``gen`` log with increasing reward and ``neg_gen`` log with reducing reward. 





#### Changes to original script
* ``imitation\src\imitation\util\utils.py`` at line 103, dropped i (needed to pass custom wrappers.)