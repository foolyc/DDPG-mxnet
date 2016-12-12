import filter_env
from ddpg import *
import gc
import gym
gc.enable()

from config import *



def main():
    env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    agent = DDPG(env)
    env.monitor.start('experiments/' + ENV_NAME, force=True)

    for episode in xrange(EPISODES):
        state = env.reset()
        # Train
        for step in xrange(env.spec.timestep_limit):
            action = agent.noise_action(state)
            next_state,reward,done,_ = env.step(action)

            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            if done:
                break
        # Testing:
        if episode % 100 == 0 and episode > 100:
            total_reward = 0
            min_reward = 1000
            max_reward = 0
            for i in xrange(TEST):
                reward_one = 0
                state = env.reset()
                for j in xrange(env.spec.timestep_limit):
                    #env.render()
                    action = agent.action(state) # direct action for test
                    state,reward,done,_ = env.step(action)
                    total_reward += reward
                    reward_one += reward
                    if done:
                        break
                min_reward = min(min_reward, reward_one)
                max_reward = max(max_reward, reward_one)
            ave_reward = total_reward/TEST
            print 'episode: ',episode,'Evaluation Reward: Average-',ave_reward, "  Min-", min_reward, "  Max-", max_reward
    env.monitor.close()

if __name__ == '__main__':
    main()
