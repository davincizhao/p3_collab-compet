# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
import torch.nn.functional as F
from utilities import soft_update, transpose_to_tensor, transpose_list,trans_np_to_tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("device:",device)
#device = 'cpu'



class MADDPG:
    def __init__(self, discount_factor=0.99, tau=0.001):
        super(MADDPG, self).__init__()

        # critic input = obs_full + actions = 48+2+2=52
        self.maddpg_agent = [DDPGAgent(24, 256, 128, 2, 26, 256, 128), 
                             DDPGAgent(24, 256, 128, 2, 26, 256, 128)]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0
    
    def get_actors(self):
        #get actors of all the agents in the MADDPG object
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        # get target_actors of all the agents in the MADDPG object
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=0.0):
        #get actions from all agents in the MADDPG object
        actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        #get target network actions from all the agents in the MADDPG object 
        target_actions = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
  
        return target_actions
    
    #def update(self, samples, agent_number, logger):
    def update(self, samples, agent_number):
        """update the critics and actors of all the agents """

        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]
        s_obs, s_action, s_reward, s_next_obs,  s_done = map(transpose_to_tensor, samples)
        
        #print("\n agent_number={},obs list size= {} * {} * {} ".format(agent_number,len(s_obs),len(s_obs[0]),len(s_obs[0][0])))
        
        obs = s_obs[agent_number]
        action = s_action[agent_number]
        rewards = s_reward[agent_number]
        next_obs = s_next_obs[agent_number]
        dones = s_done[agent_number]
        
        #obs = torch.stack(obs) 
        #print("\n agent_number={},obs list size= {} * {} * {} ".format(agent_number,len(obs),len(obs[0]),len(obs[0][0])))
        #next_obs = torch.stack(next_obs)
        #print(" obs_full_stack={}".format(obs_full.size()))
        agent = self.maddpg_agent[agent_number]
        #agent.critic_optimizer.zero_grad()

        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = agent.target_act(next_obs)
        #print(" target_actions_shape={}".format(target_actions.size()))
        #target_actions = torch.cat(target_actions, dim=1)
       
        #print(" target_actions_change_shape={}".format(target_actions.size()))
        #print("type",type(next_obs),type(target_actions))
        #target_critic_input = torch.cat((next_obs,target_actions), dim=1).to(device)
        target_critic_input = torch.cat((next_obs.to(device),target_actions.to(device)), dim=1)
        
        #print(" target_critic_input",target_critic_input)
        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)
        
        y = rewards.view(-1, 1).to(device) + (self.discount_factor * q_next.to(device) * (1 - dones.view(-1, 1).to(device)))
        #action = torch.cat(action, dim=1)
        #print(" action_size={}".format(action.size()))
        critic_input = torch.cat((obs, action), dim=1).to(device)
        #print(" critic_input={}".format(critic_input.size()))
        q = agent.critic(critic_input)

        #huber_loss = torch.nn.SmoothL1Loss()
        #critic_loss = huber_loss(q, y.detach())
        #print(" q_shape={},y={},re={},q_next={},dones={}".format(q.size(),y.size(),rewards.size(),q_next.size(),dones.size()) )
        critic_loss = F.mse_loss(q, y)
        agent.critic_optimizer.zero_grad()
        
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1)
        agent.critic_optimizer.step()


        # make input to agent
        q_input = agent.actor(obs.to(device))        
        # many of the obs are redundant, and obs[1] contains all useful information already
        q_input2 = torch.cat((obs.to(device), q_input), dim=1)
        
        # get the policy gradient
        actor_loss = -agent.critic(q_input2.to(device)).mean()
        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()       
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()
        


    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
            
            
            




