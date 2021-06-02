# Not-GAIL
To install each library here use ``pip install -e``

Make sure you are not using python 3.9, I faced many installation problems with it. Python 3.7 or 3.8 is better.

#### Use Curriculum Learning for unfair speedups!
make sure the -p flag is passed (need partially obs so cnn does not get angry)
``python .\minigrid_ppo_training_script.py -e MiniGrid-ColoredFourRooms-v0 -r partial_img_colored4rooms -p -le MiniGrid-Empty-16x16-v0 -l partial_img_empty_16x16``
Need pre-trained weights on simpler task, pass using -l and -le

#### For empty middle room 
collected traj saved in ``traj_datasets\middle_empty_random_traj``
train BaC with ``python .\test_crnn_bac_triplet.py -e MiniGrid-MidEmpty-Random-6x6-v0 -t middle_empty_random_traj --bc -s avoid_middle_6x6``
load and check BaC with ``python .\test_crnn_bac_triplet.py -e MiniGrid-MidEmpty-Random-6x6-v0 -s avoid_middle_6x6 -l``
weights saved in ``bac_weights\avoid_middle_6x6.pt``

#### Scripts

``python .\minigrid_ppo_training_script.py -e MiniGrid-Empty-Random-6x6-v0 -r cnn_1 --seed 1 --show``

``python .\minigrid_traj_collection_script.py -e MiniGrid-Empty-Random-6x6-v0 -r cnn_1 -s test_traj_collection --ntraj 20 --render``

``python .\test_crnn_bac_triplet.py -e MiniGrid-KeyEmpty-6x6-v0  -t 100_key_6x6_traj --bc -s key_6x6_bac_attn_switched -te 60 -ce 120``


#### New Minigrid Envs
* ``MiniGrid-KeyEmpty``: Similar to the empty grid with agent and key position randomized.
    * Found in : ``gym_minigrid/envs/keyempty.py``
    * Variants: 
        * ``MiniGrid-KeyEmpty-16x16-v0``
        * ``MiniGrid-KeyEmpty-8x8-v0``
        * ``MiniGrid-KeyEmpty-6x6-v0``

* ``MiniGrid-nKeyEmpty``: Similar to the key empty grid with agent and n keys randomized.
    * Found in : ``gym_minigrid/envs/nkeyempty.py``
    * Variants: 
        * ``MiniGrid-3KeyEmpty-8x8-v0``
        * ``MiniGrid-2KeyEmpty-6x6-v0``


* ``MiniGrid-MidEmpty``: Similar to the empty grid with agent avoiding middle blocks.
    * Found in : ``gym_minigrid/envs/middleempty.py``
    * Variants: 
        * ``MiniGrid-MidEmpty-6x6-v0``
        * ``MiniGrid-MidEmpty-Random-6x6-v0``

* ``MiniGrid-ColoredFourRooms``: Similar to 4 rooms env, with top right room colored as yellow
    * Found in : ``gym_minigrid/envs/colored_fourrooms.py``
    * Variants: 
        * ``MiniGrid-ColoredFourRooms-v0``





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