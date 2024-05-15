from torch import nn
import torch

class DQN(nn.Module):
    def __init__(self, n_acts):
        super(DQN, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=2), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU())
        self.layer3 = nn.Sequential( nn.Linear(1600, 256), nn.ReLU())
        self.fc = nn.Linear(256, n_acts)

    def forward(self, x):
        x =  nn.MaxPool2d(kernel_size=2)(self.layer1(x))
        x = nn.MaxPool2d(kernel_size=2)(self.layer2(x))
        x = x.view(-1, 1600)
        x = self.layer3(x)
        x = self.fc(x)
        return x
    
    def train_on_batch(self, target_model, optimizer, obs, acts, rewards, next_obs, terminals, gamma=0.99):
        next_q_values = target_model.forward(next_obs)
        max_next_q = torch.max(next_q_values, dim=1)[0].detach()

        terminal_mod = 1 - terminals
        label_q_val = rewards + terminal_mod * gamma * max_next_q

        pred_q_val = self.forward(obs)
        pred_q_val = pred_q_val.gather(index=acts.view(-1,1), dim=1).view(-1)

        loss = torch.mean((label_q_val - pred_q_val)**2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    

if __name__ == '__main__':
    model = DQN(4)
    x = torch.randn(1, 4, 100, 100)
    # this automatically calls model.forward(x) as it is implemented in the nn.Module class by implementing the __call__ method
    y = model(x)  
    print(x.shape, y)

