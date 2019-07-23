# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from time import sleep
from threading import Thread

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data

import numpy as np
import math
import os
import random
import pandas
import parameters

from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from std_msgs.msg import String
from gazebo_msgs.srv import GetEntityState, SetEntityState

class TurtlebotEnv():
    def __init__(self):
        self.node_ = rclpy.create_node('turtlebot3_env_oa')
        self.act = 0
        self.done = False
        self.actions = [[parameters.ZERO, parameters.ZERO], # Stop
                        [parameters.ZERO, -parameters.SPIN_VELOCITY], # SR
                        [parameters.STEER_FWD_VELOCITY, -parameters.STEER_VELOCITY], # SFR
                        [parameters.FWD_VELOCITY, parameters.ZERO], # FWD
                        [parameters.STEER_FWD_VELOCITY, parameters.STEER_VELOCITY], # SFL
                        [parameters.ZERO, parameters.SPIN_VELOCITY], # SL
                        [-parameters.STEER_FWD_VELOCITY, -parameters.STEER_VELOCITY], # SBL
                        [-parameters.FWD_VELOCITY, -parameters.ZERO],# BWD 
                        [-parameters.STEER_FWD_VELOCITY, parameters.STEER_VELOCITY]] # SBR                        

        self.bonous_reward = 0

        self.collision = False
        self.collision_tol = 0.125
        self.laser_scan_range = [0] * 360
        self.states_input = [3.5] * 8
        self.zero_div_tol = 0.01
        self.range_min = 0.0
        self.t_penalty = 0.0
        self.states = []

        self.current_pose = Pose()
        self.goal_pose = Pose()

        self.pub_cmd_vel = self.node_.create_publisher(Twist, 'cmd_vel', 1)
        self.sub_scan = self.node_.create_subscription(LaserScan, 'scan', self.scan_callback, qos_profile_sensor_data)

        self.reset_simulation = self.node_.create_client(Empty, 'reset_simulation')
        self.reset_world = self.node_.create_client(Empty, 'reset_world')
        self.unpause_proxy = self.node_.create_client(Empty, 'unpause_physics')
        self.pause_proxy = self.node_.create_client(Empty, 'pause_physics')
        self.get_entity_state = self.node_.create_client(GetEntityState, 'get_entity_state')
        self.set_entity_state = self.node_.create_client(SetEntityState, 'set_entity_state')
        self.scan_msg_received = False
        self.t = Thread(target=rclpy.spin, args=[self.node_])
        self.t.start()
    
    def cleanup(self):
        self.t.join()

    def get_reward(self):
        reward = 0 #should I use the pre_reward?
        goal_distance_reward = (10/self.sq_distance_to_goal())

        
        if self.collision == True:
            reward = -10 #- self.t_penalty
            self.done = True
            print('loss reward:  '+ str(reward))
            return reward, self.done

        elif self.sq_distance_to_goal() > 1:
            reward = goal_distance_reward #- self.t_penalty
            print('goal_distance: ' + str(self.sq_distance_to_goal())  + ' reward: ' + str(reward))
        else:
            reward = 150
            print('win reward:  '+ str(reward))
            self.done = True
        self.t_penalty += 0.005
        return reward, self.done


    def scan_callback(self, LaserScan):
        self.scan_msg_received = True
        self.laser_scan_range = []
        self.range_min = LaserScan.range_min
        range_max = LaserScan.range_max

        for i in range(len(LaserScan.ranges)):
            if LaserScan.ranges[i] == float('Inf'):
                self.laser_scan_range.append(range_max)
            elif LaserScan.ranges[i] < self.range_min:
                self.laser_scan_range.append(self.range_min + self.collision_tol)
            else:
                self.laser_scan_range.append(LaserScan.ranges[i])
        if self.check_collision():
            self.collision = True
            self.done = True
        self.states_input = []
        for i in range(8):
            step = int(len(LaserScan.ranges)/8)
            self.states_input.append(min(self.laser_scan_range[i*step:(i+1)*step], default=0))

    def action_space(self):
        return len(self.actions)

    def step(self, action):
        vel_cmd = Twist()
        self.act = action
        vel_cmd.linear.x = self.actions[action][0]
        vel_cmd.angular.z = self.actions[action][1]
        self.pub_cmd_vel.publish(vel_cmd)
        vel_cmd.linear.x = 0.0
        vel_cmd.angular.z = 0.0
        sleep(parameters.LOOP_RATE)
        get_reward = self.get_reward()
        return self.observation(), get_reward[0], self.done
    
    def observation(self):
        self.get_robot_pose()
        self.states.clear()
        self.states = [0]*12
        self.states[0:7] = self.states_input
        self.states[8] = float(self.current_pose.position.x)
        self.states[9] = float(self.current_pose.position.y)        
        self.states[10] = float(self.goal_pose.position.x)
        self.states[11] = float(self.goal_pose.position.y)
        self.states[12] = self.sq_distance_to_goal()
        return  self.states

    def check_collision(self):
        if  min(self.laser_scan_range) < self.range_min + self.collision_tol:
            print("Near collision detected... " + str(min(self.laser_scan_range)))
            return True
        return False

    def stop_action(self):
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.0
        vel_cmd.angular.z = 0.0
        self.pub_cmd_vel.publish(vel_cmd)

    def set_random_robot_pose(self):
        sleep(1.0)
        while not self.set_entity_state.wait_for_service(timeout_sec=1.0):
             print('Set entity state service is not available...')
        random_pose = self.get_random_pose()
        req = SetEntityState.Request()
        req.state.name = 'turtlebot3_waffle'
        req.state.pose.position.x = random_pose.position.x
        req.state.pose.position.y = random_pose.position.y
        req.state.pose.position.z = 0.0
        req.state.pose.orientation.x = random_pose.orientation.x
        req.state.pose.orientation.y = random_pose.orientation.y
        req.state.pose.orientation.z = random_pose.orientation.z
        req.state.pose.orientation.w = random_pose.orientation.w
        future = self.set_entity_state.call_async(req)

        while not future.done() and rclpy.ok():            
            sleep(0.1)
        sleep(1.0)
    
    def get_robot_pose(self):
        while not self.get_entity_state.wait_for_service(timeout_sec=1.0):
            print('get entity state service is not available...')
        req = GetEntityState.Request()
        req.name = 'turtlebot3_waffle'
        future = self.get_entity_state.call_async(req)

        while not future.done() and rclpy.ok():            
            sleep(0.01)

        self.current_pose.position.x = future.result().state.pose.position.x
        self.current_pose.position.y = future.result().state.pose.position.y
        self.current_pose.position.z = future.result().state.pose.position.z
        self.current_pose.orientation.x = future.result().state.pose.orientation.x
        self.current_pose.orientation.y = future.result().state.pose.orientation.y
        self.current_pose.orientation.z = future.result().state.pose.orientation.z
        self.current_pose.orientation.w = future.result().state.pose.orientation.w

    def get_random_pose(self):
        random_pose = Pose()
        validPoseFile = os.getenv('VALID_POSE')
        df = pandas.read_csv(validPoseFile, header=0, names=['x', 'y'])
        nrows = df['x'].count()

        rand_index = random.randint(1, nrows)
        yaw = random.uniform(0,math.pi*2)

        random_pose.position.x = df['x'][rand_index-1]
        random_pose.position.y = df['y'][rand_index-1]
        random_pose.position.z = 0.0
        random_pose.orientation.x = 0.0
        random_pose.orientation.y = 0.0
        random_pose.orientation.z = math.sin(yaw*0.5)
        random_pose.orientation.w = math.cos(yaw*0.5)

        return random_pose
    
    def sq_distance_to_goal(self):
        self.get_robot_pose()
        dx = self.goal_pose.position.x - self.current_pose.position.x
        dy = self.goal_pose.position.y - self.current_pose.position.y
        return dx * dx + dy * dy

    def reset(self):
        self.t_penalty = 0.0
        self.scan_msg_received = False
        self.stop_action    
        while not self.reset_world.wait_for_service(timeout_sec=1.0):
            print('Reset world service is not available...')
        self.reset_world.call_async(Empty.Request())

        while not self.reset_simulation.wait_for_service(timeout_sec=1.0):
            print('Reset simulation service is not available...')
        self.reset_simulation.call_async(Empty.Request())

        self.set_random_robot_pose()
        self.get_robot_pose()
        self.goal_pose = self.get_random_pose()

        self.laser_scan_range = [0] * 360
        self.states_input = [3.5] * 8
        while not self.scan_msg_received and rclpy.ok():
            sleep(0.1)
        self.collision = False
        self.done = False
        self.bonous_reward = 0

        return self.observation()