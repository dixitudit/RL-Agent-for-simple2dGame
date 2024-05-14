import torch
import numpy as np
from main1 import SpaceInvaderAI
import random
import cv2
from model import DQN

class ExperienceReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    # stores the state, action, reward, next_state, done
    def add_step(self, step):
        self.memory.append(step)
        if len(self.memory) > self.capacity:
            self.memory = self.memory[-self.capacity:]
     
    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        # Randomly sample batch_size elements from memory
        samples = np.random.choice(len(self.memory), batch_size, replace=False)
        
        # Separate components
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for idx in samples:
            state, action, reward, next_state, done = self.memory[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        # Convert lists to NumPy arrays
        states = torch.tensor(np.array(states)).float()
        actions = torch.tensor(np.array(actions)).long()
        rewards = torch.tensor(np.array(rewards)).float()
        next_states = torch.tensor(np.array(next_states)).float()
        dones = torch.tensor(np.array(dones)).float()
        
        return states, actions, rewards, next_states, dones


class Agent:
    def __init__(self, n_acts):
        self.n_games = 0
        
    # preprocess the image
    def preprocess(self, state, prev_frame):
        state = self.filter_obs(state)
        # cv2.imshow('state',state)
        state, prev_frame = self.stacked_frames(state, prev_frame)
        # cv2.imshow('frame1', prev_frame[0])
        # cv2.imshow('frame2', prev_frame[1])
        # cv2.imshow('frame3', prev_frame[2])
        
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return state, prev_frame

    def format_reward(self, reward):
        if reward > 0:
            return 1
        elif reward < 0:
            return -1
        else:
            return 0

    # downsample the image to 300x300 and convert to grayscale and normalize
    def filter_obs(self, obs):
        obs = cv2.resize(obs, (100, 100), interpolation=cv2.INTER_LINEAR)
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        obs = obs / 255.0

        return obs
    # stack the frames for model to understand the movement
    def stacked_frames(self, state, prev_frame):
        if len(prev_frame) == 0:
            prev_frame = [state]*3
        prev_frame.append(state)
        stacked_frames = np.stack(prev_frame)
        prev_frame = prev_frame[1:]

        return stacked_frames, prev_frame
    

def save_checkpoint(epoch, model, optimizer, loss):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, 'checkpoint.pth')

# Load the model checkpoint and resume training
def load_checkpoint(model, optimizer):
    checkpoint = torch.load('checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

if __name__ == '__main__':
    agent = Agent(3)
    n_acts = 4
    model = DQN(n_acts)
    game = SpaceInvaderAI()
    learning_rate = 2.5e-4
    update_freq = 4
    prev_frame = []
    episodes = 10000
    max_steps = 1000
    ExReplayCapacity = 50000
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, eps=1e-6)
    frame_skip = 3
    all_rewards = []
    global_step = 0
    epsilon = lambda step: np.clip(1-0.9*(step/1e5), 0.1, 1)
    er = ExperienceReplay(ExReplayCapacity)
    # epoch = 0
    model, optimizer, epoch, loss = load_checkpoint(model, optimizer)


    for episode in range(episodes):
        game = SpaceInvaderAI()
        prev_frame = []
        obs, prev_frame = agent.preprocess(game.draw_game_window(), prev_frame)
        episode_reward = 0
        step = 0
        while step < max_steps:
            if random.random() < epsilon(global_step+epoch):
                action = random.randint(0, n_acts-1)
            else:
                state = torch.tensor(obs).unsqueeze(0).float()
                q_values = model(state)[0]
                action = torch.argmax(q_values)

            cumulated_reward = 0

            for _ in range(frame_skip):
                reward, done, score = game.play_step(action)
                next_obs, prev_frame = agent.preprocess(game.draw_game_window(), prev_frame)
                cumulated_reward += reward
                if done or step >= max_steps:
                    break
                        
            
            episode_reward += cumulated_reward
            reward = agent.format_reward(cumulated_reward)
            er.add_step([obs, action, reward, next_obs, int(done)])
            obs = next_obs

    
            # if global_step % update_freq == 0:
            #     state_data, action_data, reward_data, next_state_data, done_data = er.sample(32)
            #     loss = model.train_on_batch(optimizer, state_data, action_data, reward_data, next_state_data, done_data)
            #     # print('Model training...')
            #     if global_step % 40 == 0:
            #         # print('Model updated')
            #         # save model
            #         # print(epoch)
            #         save_checkpoint(global_step+epoch , model, optimizer, loss)

            
            global_step += 1
            step += 1
            if done:
                break

        all_rewards.append(episode_reward)
        if global_step % 10==0:
            print(f'Epoch: {global_step+epoch},Episode: {episode}, Reward: {episode_reward}, Epsilon: {epsilon(global_step+epoch)}, avg_rewards: {np.mean(all_rewards[-10:])}')

