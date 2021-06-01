from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class KeyEmptyEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    Key and Agent position randomized
    """

    def __init__(
        self, size=8,
    ):
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
        
        self.key = Key(color="green")
        # Place a key randomly
        self.place_obj(self.key)

        # Place the agent in a random position

        self.place_agent()

        self.mission = "pickup the key"
    
    def step(self, action):
        obs, reward, done, info = super().step(action)

        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.key:
                reward = self._reward()
                done = True

        return obs, reward, done, info

class KeyEmptyEnv6x6(KeyEmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=6, **kwargs)


class KeyEmptyEnv8x8(KeyEmptyEnv):
    def __init__(self):
        super().__init__(size=8)


class KeyEmptyEnv16x16(KeyEmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=16, **kwargs)


register(
    id="MiniGrid-KeyEmpty-16x16-v0", entry_point="gym_minigrid.envs:KeyEmptyEnv16x16"
)

register(id="MiniGrid-KeyEmpty-6x6-v0", entry_point="gym_minigrid.envs:KeyEmptyEnv6x6")

register(id="MiniGrid-KeyEmpty-8x8-v0", entry_point="gym_minigrid.envs:KeyEmptyEnv8x8")

