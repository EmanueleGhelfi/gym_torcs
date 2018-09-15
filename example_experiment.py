from gym_torcs import TorcsEnv
from snakeoil_agent import Agent
import numpy as np

episode_count = 10
max_steps = 10000
reward = 0
done = False
step = 0


# Generate a Torcs environment
env = TorcsEnv(vision=False, throttle=True, return_np=False)

agent = Agent(2)  # steering only

print("TORCS Experiment Start.")
for i in range(episode_count):
    print("Episode : " + str(i))

    if np.mod(i, 3) == 0:
        # Sometimes you need to relaunch TORCS because of the memory leak error
        ob = env.reset(relaunch=True)
    else:
        ob = env.reset()
    total_reward = 0.
    for j in range(max_steps):
        action = agent.act(ob, reward, done, False)

        ob, reward, done, info = env.step(action)
        total_reward += reward

        step += 1
        if done:
            print(info)
            break

    print("TOTAL REWARD @ " + str(i) +" -th Episode  :  " + str(total_reward))
    print("Total Step: " + str(step))
    print("")

env.end()  # This is for shutting down TORCS
print("Finish.")
