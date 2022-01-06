import os

from gym.envs.registration import register

with open(os.path.join(os.path.dirname(__file__), "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

register(
    id="ToyPickPlaceOneObject-v1",
    entry_point="toy_gym.envs:ToyPickPlaceOneObject",
    max_episode_steps=200,
)