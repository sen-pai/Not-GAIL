from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class nKeyEmptyEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    Key and Agent position randomized
    """

    def __init__(
        self, size=8, keys = 2
    ):
        self.nkeys = keys
        super().__init__(
            grid_size=size,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=True,
        )

        

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        self.keys = []
        for key in range(self.nkeys):
            k = Key(color="green")
            # Place a key randomly
            self.place_obj(k)
            self.keys.append(k)
            

        # Place the agent in a random position

        self.place_agent()

        self.mission = "pickup the key"
    
    def step(self, action):
        obs, reward, done, info = super().step(action)

        if action == self.actions.pickup:
            if self.carrying and self.carrying in self.keys:
                reward = self._reward()
                done = True

        return obs, reward, done, info

class KeyEmptyEnv6x6_2(nKeyEmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=6, keys = 2, **kwargs)


class KeyEmptyEnv8x8_3(nKeyEmptyEnv):
    def __init__(self):
        super().__init__(size=8, keys = 3)



register(id="MiniGrid-2KeyEmpty-6x6-v0", entry_point="gym_minigrid.envs:KeyEmptyEnv6x6_2")

register(id="MiniGrid-3KeyEmpty-8x8-v0", entry_point="gym_minigrid.envs:KeyEmptyEnv8x8_3")

