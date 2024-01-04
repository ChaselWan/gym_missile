# -*- encoding: utf-8 -*-
"""
@File       :   reward_settings.py
@Contact    :   chasel_wan@nuaa.edu.cn
@Author     :   Chasel Wan
@Time       :   2023/12/02
@Version    :   1.0
@Desciption :   A gym wrapper for missile guidance
"""

import time
import gym
import math
import numpy as np
from matplotlib import pyplot as plt
import time
import random
from reward_settings import reward_params


class MissileEnv(gym.Env):

    def __init__(self, params):
        self.save_times = 0
        self.timesteps = 0
        self.Eval_mode = params['Eval_mode']
        self.params = params
        
        # global variable
        self.pi = 3.14
        self.timestep = 0.1
        self.P = 7000 # thrust
        self.g = 9.81
        self.Fx = 200 # resistance
        self.Fy = 250 # lift
        self.m =55 # mass
        self.alpha = 0.1 # angle of attack
        self.Vpn = 2000 # Proportional guidance terminal speed under the same conditions
        self.done = False
        self.handover = False
        self.handover_x = None
        self.handover_y = None
        self.handover_v = None
        self.handover_theta = None
        self.state_demo = np.random.uniform(low=-0.05, high=0.05, size=(7,))

        # External parameters
        self.initial_target_x = params["initial_target_x"]
        self.initial_target_y = params["initial_target_y"]
        self.thetaf = params["initial_thetaf"] * self.pi / 180
        self.initial_missile_x = params["initial_missile_x"]
        self.initial_missile_y = params["initial_missile_y"]
        self.initial_velocity = params["initial_velocity"]
        self.initial_theta = params["initial_theta"] * self.pi / 180
        self.PN_param = params["PN_param"]
        self.PN_handover_r = params["PN_handover_r"]
        self.render_figure_size = params["render_figure_size"]
        self.interception_range = params["interception_range"]
        self.move = params["target_maneuver_mode"]
        self.target_move_theta = params["target_move_theta"] * self.pi / 180
        self.r_target_move = params["target_move_r"]
        self.times_target_move = params["target_move_times"]
        self.target_move_variable = params["target_move_range"]
        if self.times_target_move > 0:
            self.average_r_target_move = self.r_target_move / self.times_target_move
        if self.times_target_move == 0:
            self.r_target_move = 0
            self.average_r_target_move = 0
        
        # Store application data
        self.target_trajectory_x = []
        self.target_trajectory_y = []  
        self.missile_trajectory_x = []
        self.missile_trajectory_y = []
        self.velocity = []
        self.q = None
        self.q_list = []
        self.theta = None
        self.a = []
        self.r = None
        self.r_list = []
        self.eta = None
        self.etaf = None
        self.epsilon = []
        self.theta_list = []
        self.rel_velocity = None
        self.rel_velocity_list = []

    def reset(self):
        # Initialize trajectory and starting point and end point
        self.timestep = 0.1
        self.timesteps = 0
        self.save_times += 1
        self.done = False
        self.handover = False
        self.missile_trajectory_x = []
        self.missile_trajectory_y = []
        self.velocity = []
        self.rel_velocity = None
        self.rel_velocity_list = []
        self.k = [0]
        
        self.missile_trajectory_x.append(self.initial_missile_x)
        self.missile_trajectory_y.append(self.initial_missile_y)
        self.velocity.append(self.initial_velocity)

        self.a = []
        self.r_list = []
        self.theta_list = []
        
        self.q_list = []
        self.target_trajectory_x = []
        self.target_trajectory_y = []
        self.target_x = self.initial_target_x
        self.target_y = self.initial_target_y
        self.target_trajectory_x.append(self.target_x)
        self.target_trajectory_y.append(self.target_y)
        self.a.append(0)
        self.r_target_move = self.params["target_move_r"]
        self.thetaf = self.params["initial_thetaf"] * self.pi / 180
        print("Target position is [{},{}], Terminal angle is：{}".format(self.target_x, self.target_y, self.thetaf * 180 / self.pi))
        
        # Calculate the initial value of some variables
        self.q = math.atan(
            (self.target_y - self.initial_missile_y) / (self.target_x - self.initial_missile_x))
        self.q_list.append(self.q)
        self.r = math.sqrt((self.initial_missile_x - self.target_x) ** 2 + (
                self.initial_missile_y - self.target_y) ** 2)
        self.r_list.append(self.r)
        self.theta = self.initial_theta
        self.rel_velocity = self.velocity[-1] * math.cos(self.thetaf-self.theta)
        self.rel_velocity_list.append(self.rel_velocity)
        self.theta_list.append(self.theta)
        self.eta = self.q - self.theta
        self.etaf = self.thetaf - self.theta

        self.state = [self.initial_missile_x,
                      self.initial_missile_y,
                      self.initial_velocity*math.cos(self.thetaf-self.theta),
                      self.thetaf-self.theta,
                      self.k[-1],
                      self.target_x,
                      self.target_y]

        self.state = np.array(self.state)
        self.state = self.state.reshape(self.state_demo.shape)
     
        return np.array(self.state, dtype=np.float32),{}

    def step(self, action):
    
        k = action[0] * 5 + 3
        epsilon = (action[1] - 0.5) * 2 * 0.1
        self.epsilon.append(epsilon)
        self.k.append(k)
        if self.r < self.r_target_move:  # target maneuver
            if self.move == "uniform":
                self.target_x += random.uniform(-1,1) * self.target_move_variable
                self.target_y += random.uniform(-1,1) * self.target_move_variable
                self.target_trajectory_x.append(self.target_x)
                self.target_trajectory_y.append(self.target_y)
                self.r_target_move -= self.average_r_target_move
                self.thetaf += random.uniform(-1,1) * self.target_move_theta
                
            if self.move == "gauss":
                self.target_x += np.array(random.gauss(0,1)).clip(-1,1) * self.target_move_variable
                self.target_y += np.array(random.gauss(0,1)).clip(-1,1) * self.target_move_variable
                self.target_trajectory_x.append(self.target_x)
                self.target_trajectory_y.append(self.target_y)
                self.r_target_move -= self.average_r_target_move
                self.thetaf += np.array(random.gauss(0,1)).clip(-1,1) * self.target_move_theta
            if not self.Eval_mode:
                print("The position of the target after maneuver is：[{},{}],The terminal angle after maneuvering is：{}".format(self.target_x, self.target_y, self.thetaf *180 / self.pi))
        missile_x, missile_y, velocity = self._get_missile_move(k, epsilon)

        self.missile_trajectory_x.append(missile_x)
        self.missile_trajectory_y.append(missile_y)
        self.velocity.append(velocity)
        self.rel_velocity = velocity*math.cos(self.thetaf-self.theta)
        self.rel_velocity_list.append(self.rel_velocity)
        self.state = [missile_x,
                      missile_y,
                      velocity*math.cos(self.thetaf-self.theta),
                      self.thetaf-self.theta,
                      self.k[-1],
                      self.target_x,
                      self.target_y]
        self.state = np.array(self.state)
        self.state = self.state.reshape(self.state_demo.shape)
        # if self.Eval_mode:
        #     self.render()
        
        if self.r < self.PN_handover_r:  # forecast correction
            self.handover_x = self.missile_trajectory_x[-1]
            self.handover_y = self.missile_trajectory_y[-1]
            self.handover_v = self.velocity[-1]
            self.handover_theta = self.theta
            if not self.Eval_mode:
                print("Arrive at the handover point{}.Start mid-term and end-of-term shift handovet.target position is{}.The target angle is{}.The handover speed is{}.The handover angle is{}".format([self.handover_x, self.handover_y], [self.target_x, self.target_y], self.thetaf * 180 / self.pi, self.handover_v, self.handover_theta * 180 / self.pi))
            if not self.Eval_mode:
                time.sleep(1)
            self.handover = True
            if not self.Eval_mode:
                print("Now starts terminal guidance with conventional proportional guidance with coefficient {}".format(k))
            self._terminal_guide(k, velocity)
            if not self.Eval_mode:
                print("Terminal guidance ends, the strike angle is: {}, the strike speed is: {}, and the relative speed of the projectile is: {}".format(self.theta * 180 / self.pi, self.velocity[-1], self.velocity[-1] * math.cos(self.thetaf - self.theta)))

        if self.theta > 1.57 or self.theta < 0:
            self.done = True
            # print("The ballistic inclination angle changes too much and the round is terminated.")
        if self.done:
            print("The step size of this round is", self.timesteps)
        r = self._get_reward()
        return np.array(self.state, dtype=np.float32), r, self.done, {}

    def render_terminal(self):
        plt.gcf().clf()
        plt.plot(self.missile_trajectory_x, self.missile_trajectory_y, linewidth = 1.5)
        for i in range(len(self.target_trajectory_x)):
            if i == 0:
                plt.plot(self.target_trajectory_x, self.target_trajectory_y, 'o', markersize = 5, c = 'r')
                plt.text(self.target_trajectory_x[i], self.target_trajectory_y[i], (self.target_trajectory_x[i], self.target_trajectory_y[i]))
            if i == 1:
                plt.plot(self.target_trajectory_x, self.target_trajectory_y, 'o', markersize = 5, c = 'g')
                plt.text(self.target_trajectory_x[i], self.target_trajectory_y[i], (self.target_trajectory_x[i], self.target_trajectory_y[i]))
        # plt.plot(self.target_trajectory_x[-1], self.target_trajectory_y[-1], 'o', markersize = 5)
        # for i in range(len(self.target_trajectory_x)):
        #     plt.text(self.target_trajectory_x[i], self.target_trajectory_y[i], (self.target_trajectory_x[i], self.target_trajectory_y[i]), color='r')
        if self.handover_x:
            plt.plot(self.handover_x, self.handover_y, '<', markersize = 7.5, c = 'r')
            plt.text(self.handover_x, self.handover_y, "handover_point")
        plt.show(block=False)
        plt.pause(0.05)
    
    def render(self, episode, episode_return):
        plt.figure(figsize = self.render_figure_size)
        plt.plot(self.missile_trajectory_x, self.missile_trajectory_y, linewidth = 1.5)
        for i in range(len(self.target_trajectory_x)):
            if i == 0:
                plt.plot(self.target_trajectory_x[i], self.target_trajectory_y[i], 'o', markersize = 5, c = 'r')
                plt.text(self.target_trajectory_x[i], self.target_trajectory_y[i], (self.target_trajectory_x[i], self.target_trajectory_y[i]))
            if i == 1:
                plt.plot(self.target_trajectory_x[i], self.target_trajectory_y[i], 'o', markersize = 5, c = 'g')
                plt.text(self.target_trajectory_x[i], self.target_trajectory_y[i], (self.target_trajectory_x[i], self.target_trajectory_y[i]))
        plt.savefig('new_trajectory_figure/episode_{}_return_{}.png'.format(episode, episode_return))
    
    def close(self):
        plt.pause(0.1)
        plt.close()

    def _get_reward(self):
        reward_step = 0
        reward_theta = 0
        reward_boundary = 0
        reward_handover = 0
        reward_terminal_theta = 0
        reward_interception = 0
        reward_velocity = 0
        reward_energy = 0
        reward_r = self.r_list[-2] - self.r_list[-1]
        reward_energy = 0.1 - abs(self._get_d_a())
        

        if self.q < self.thetaf and self.q >= self.thetaf - 5*self.pi/180:
            reward_theta = self.thetaf - self.q
        else:
            reward_theta = - abs(self.thetaf - self.q)

        
        if self.done:
            if self.r < 4000 and not self.handover:
                reward_handover = -20
            if self.handover:
                reward_handover = 20
                reward_r = 0

                if self.r_list[-2] <= 10*self.interception_range and self.r_list[-2] > 9*self.interception_range:
                    reward_interception += 1
                if self.r_list[-2] <= 9*self.interception_range and self.r_list[-2] > 8*self.interception_range:
                    reward_interception += 2
                if self.r_list[-2] <= 8*self.interception_range and self.r_list[-2] > 7*self.interception_range:
                    reward_interception += 3
                if self.r_list[-2] <= 7*self.interception_range and self.r_list[-2] > 6*self.interception_range:
                    reward_interception += 4
                if self.r_list[-2] <= 6*self.interception_range and self.r_list[-2] > 5*self.interception_range:
                    reward_interception += 5
                if self.r_list[-2] <= 5*self.interception_range and self.r_list[-2] > 4*self.interception_range:
                    reward_interception += 6
                if self.r_list[-2] <= 4*self.interception_range and self.r_list[-2] > 3*self.interception_range:
                    reward_interception += 7
                if self.r_list[-2] <= 3*self.interception_range and self.r_list[-2] > 2*self.interception_range:
                    reward_interception += 8
                if self.r_list[-2] <= 2*self.interception_range and self.r_list[-2] > self.interception_range:
                    reward_interception += 9
                if self.r_list[-2] <= self.interception_range:
                    reward_interception += 10
                reward_terminal_theta = 5 - abs(self.thetaf-self.theta) * 180 / self.pi # 1 附近
                reward_velocity = self.velocity[-1]*math.cos(self.thetaf-self.theta) / self.Vpn - 1
                
            if self.theta > 1.57 or self.theta < 0:
                reward_boundary = -200
                reward_r = 0
        
        reward = reward_params["reward_energy"] * reward_energy + reward_params["reward_theta"] * reward_theta + reward_params["reward_velocity"] * reward_velocity + reward_params["reward_interception"] * reward_interception + reward_params["reward_r"] * reward_r + reward_params["reward_boundary"] * reward_boundary + reward_params["reward_step"] * reward_step + reward_params["reward_handover"] * reward_handover + reward_params["reward_terminal_theta"] * reward_terminal_theta
        if not self.Eval_mode:
            print("The current rewards are: step reward{}, overload reward{}, angle reward{}, interception reward{}, speed reward{}, distance reward{}, boundary reward{}, shift reward{}, terminal angle reward{}".format(reward_step, reward_params["reward_energy"] * reward_energy,reward_params["reward_theta"] * reward_theta, reward_params["reward_interception"] * reward_interception, reward_params["reward_velocity"] * reward_velocity, reward_params["reward_r"] * reward_r, reward_params["reward_boundary"] * reward_boundary, reward_handover, reward_params["reward_terminal_theta"] * reward_terminal_theta))
        return reward

    def _get_missile_move(self, k, epsilon):
        missile_x_present = self.missile_trajectory_x[-1]
        missile_y_present = self.missile_trajectory_y[-1]
        v_present = self.velocity[-1]

        qdot = v_present * math.sin(self.eta) / self.r
        if not self.Eval_mode:
            print("The line of sight angular rate is", qdot)
        a = k * qdot + epsilon
        self.a.append(a)

        vdot = (self.P * (1 - 0.5 * self.alpha * self.alpha) - self.Fx)/self.m - self.g * math.sin(self.theta)
        thetadot = a
        xdot = v_present * math.cos(self.theta)
        ydot = v_present * math.sin(self.theta)

        missile_x_present += xdot * self.timestep
        missile_y_present += ydot * self.timestep

        rel_x = self.target_x - missile_x_present
        rel_y = self.target_y - missile_y_present
        self.q = math.atan(rel_y / rel_x)
        self.q_list.append(self.q)
        self.r = math.sqrt(rel_x * rel_x + rel_y * rel_y)
        self.r_list.append(self.r)
        self.theta += thetadot * self.timestep
        self.theta_list.append(self.theta)
        self.eta = self.q - self.theta
        self.etaf = self.thetaf - self.theta
        v_present += vdot * self.timestep

        return missile_x_present, missile_y_present, v_present
        
        
    def _get_d_theta(self):
        return self.theta_list[-1] - self.thetaf
        
    def _get_d_r(self):
        return self.r_list[-1] - self.r_list[-2]
     
    def _get_d_a(self):
        return self.a[-1] - self.a[-2]
    
    def _get_d_q(self):
        return self.q_list[-1] - self.q_list[-2]
        
    def _terminal_guide(self, PN_param, v_present):
        missile_x_present = self.missile_trajectory_x[-1]
        missile_y_present = self.missile_trajectory_y[-1]
        qdot = v_present * math.sin(self.eta) / self.r
        a = PN_param * qdot
        t = 0
        plt.figure(1)
        while True:
            vdot = (self.P * (1 - 0.5 * self.alpha * self.alpha) - self.Fx)/self.m - self.g * math.sin(self.theta)
            thetadot = a
            xdot = v_present * math.cos(self.theta)
            ydot = v_present * math.sin(self.theta)
            missile_x_present = missile_x_present + xdot * self.timestep
            missile_y_present = missile_y_present + ydot * self.timestep
            rel_x = self.target_x - missile_x_present
            rel_y = self.target_y - missile_y_present
            self.q = math.atan(rel_y / rel_x)
            self.r = math.sqrt(rel_x * rel_x + rel_y * rel_y)
            self.r_list.append(self.r)
            self.theta += thetadot * self.timestep
            self.theta_list.append(self.theta)
            self.eta = self.q - self.theta
            self.etaf = self.thetaf - self.theta
            v_present += vdot * self.timestep
            qdot = v_present * math.sin(self.eta) / self.r
            a = PN_param * qdot
            self.missile_trajectory_x.append(missile_x_present)
            self.missile_trajectory_y.append(missile_y_present)
            if self.r_list[-1] > self.r_list[-2]:
                self.theta_list.pop(-1)
                self.theta = self.theta_list[-1]
                self.rel_velocity = self.velocity[-1] * math.cos(self.thetaf-self.theta)
                self.rel_velocity_list.append(self.rel_velocity)
                print("At the end of the round, the projectile distance is: {}, the strike speed is {}, and the strike angle is {}".format(self.r_list[-2], self.velocity[-1], self.theta * 180 / self.pi))
                self.done = True
                if self.r_list[-2] <= self.interception_range:
                    print("Interception successful")
                else:
                    print("Interception failed")
                plt.close()
                break

