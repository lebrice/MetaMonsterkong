import pygame
from meta_monsterkong.envs import *
import numpy as np
import h5py

pygame.init()
# Instantiate the Game class and run the game
game = meta_monsterkong()
game.reward_type = 'Sparse'
game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
game.clock = pygame.time.Clock()
game.rng = np.random.RandomState(24)
game.init()

from PIL import Image

def get_image(game_state):
    image_rotated = np.fliplr(
        np.rot90(game_state.getScreenRGB(), 3))  # Hack to fix the rotated image returned by ple
    return image_rotated

obs_type = 'state'
obs_shape = [142]
# obs_shape = [210, 150, 3]
obs_ph = [":" for i in obs_shape]

obss = []
terminals = []
actions = []
import datetime

dt = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
if not os.path.isdir("expertruns"):
    os.mkdir("expertruns")
f = h5py.File(F"expertruns/expertrun {dt}.hdf5", "w")
# f = h5py.File(F"expertruns/expertrun.hdf5", "w")
# f["numEpisodes"] = 0
obss_ds = f.create_dataset("obss", shape=(1, *obs_shape), maxshape=(None, *obs_shape), dtype='int')
terminals_ds = f.create_dataset("terminals", shape=(1,), maxshape=(None,))
actions_ds = f.create_dataset("actions", shape=(1,), maxshape=(None,))

actCount = 0
batch_size = 50


def get_obs(game, obs_type='image'):
    if obs_type == 'image':
        return get_image(game)
    elif obs_type == 'state':
        return game.get_state()
    else:
        raise Exception('No obs_type specified')

while True:
    dt = game.clock.tick_busy_loop(30)
    obs = get_obs(game, obs_type=obs_type)
    keyPress = game.step(dt)
    key2Act = {
        97: 0,
        100: 1,
        32: 2,
        119: 3,
        115: 4,
        None: 5
    }
    act = key2Act[keyPress]

    # im = Image.fromarray(np.uint8(state))
    # im.save("test.png", format="png")
    if game.game_over():
        actCount += 1
        print(act)
        obss.append(obs)
        actions.append(act)
        terminals.append(1)

        prev_maxptr = obss_ds.shape[0]
        obss_ds.resize(prev_maxptr + len(obss), axis=0)
        terminals_ds.resize(prev_maxptr + len(obss), axis=0)
        actions_ds.resize(prev_maxptr + len(obss), axis=0)
        print("obs len" + str(len(obss)))

        # obss_ds[prev_maxptr:, :, :, :] = np.array(obss)
        obss_ds[prev_maxptr:, :] = np.array(obss).astype(int)
        terminals_ds[prev_maxptr:] = np.array(terminals)
        actions_ds[prev_maxptr:] = np.array(actions)
        actCount = 0
        obss = []
        terminals = []
        actions = []

        # f["numEpisodes"][()] += 1
        game.reset()
    else:
        if act != 5:
            actCount += 1
            print(act)
            print("Player Position" + str(game.currentBoard.Players[0].getPosition()))
            print("Princess Position" + str(game.currentBoard.Allies[0].getPosition()))
            obss.append(obs)
            actions.append(act)
            terminals.append(0)
    if actCount % batch_size == 0 and actCount >= batch_size:
        print("act count " + str(actCount))
        prev_maxptr = obss_ds.shape[0]
        obss_ds.resize(prev_maxptr + batch_size, axis=0)
        terminals_ds.resize(prev_maxptr + batch_size, axis=0)
        actions_ds.resize(prev_maxptr + batch_size, axis=0)
        print("obs len" + str(len(obss)))
        # obss_ds[prev_maxptr:, :, :, :] = np.array(obss)
        obss_ds[prev_maxptr:, :] = np.array(obss).astype(int)
        terminals_ds[prev_maxptr:] = np.array(terminals)
        actions_ds[prev_maxptr:] = np.array(actions)
        actCount = 0
        obss = []
        terminals = []
        actions = []
    pygame.display.update()