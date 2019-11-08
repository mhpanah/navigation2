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

from turtlebot3_environment import Turtlebot3Environment

import numpy as np
from math import pi, atan2, sin, cos
import math
import random
from time import sleep
import copy

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from nav2_msgs.action import ComputePathToPose
import nav2_msgs
from geometry_msgs.msg import PoseStamped, TransformStamped
from geometry_msgs.msg import Twist, Pose
from action_msgs.msg import GoalStatus

from gym import spaces

class NavigationTaskEnv(Turtlebot3Environment):
    def __init__(self):
        super().__init__()
        self.NavigationTaskEnv = NavigationTaskEnv

        self.act = 0
        self.done = False
        self.actions = self.get_actions()
        self.collision = False
        self.collision_tol = 0.125
        self.laser_scan_range = [0] * 360
        self.laser_scans = [3.5] * 8
        self.zero_div_tol = 0.01
        self.range_min = 0.0
        self.current_pose = Pose()
        self.goal_pose = Pose()

        self.result_path = None
        self.action_client_ = ActionClient(self.node_, ComputePathToPose, 'ComputePathToPose') 
        self.new_path_received = False
        self.path_is_valid = False

        self.pub_checkpoints = self.node_.create_publisher(MarkerArray, 'visualization_marker', 1)

        self.path = []
        self.path_index = 0
        self.path_resolution = 10
        self.reached_point = False
        self.checkpoint_index = 0
        self.hard_reset = True
        self.path_fail_count = 0
        self.local_path_index = 0
        self.num_check_points = 5
        self.checkpoints = [[0.0,0.0]] * self.num_check_points
        self.distance_to_checkpoint = [0.0, 0.0, 0.0, 0.0, 0.0]

        _high = [3.5]*len(self.laser_scans)
        _high.extend([pi, 10.0, 10.0])
        _high.extend([10.0]*self.num_check_points)

        _min = [3.5]*len(self.laser_scans)
        _min.extend([-pi, -10.0, -10.0])
        _min.extend([-10.0]*self.num_check_points)

        obs = self.observation()
        # self.observation_space = spaces.Dict(dict(
        #     desired_goal=spaces.Box(low=-10.0, high=10.0, shape=obs['desired_goal'].shape, dtype=np.float32),
        #     achieved_goal=spaces.Box(low=-10.0, high=10.0, shape=obs['achieved_goal'].shape, dtype=np.float32),
        #     observation=spaces.Box(low=-10.0, high=10.0, shape=obs['observation'].shape, dtype=np.float32),
        # ))
        self.observation_space = spaces.Box(np.array(_min), np.array(_high), dtype=np.float32)
        
    def send_goal(self, goal_pose):
        goal_msg = nav2_msgs.action.ComputePathToPose.Goal()
        goal_msg.pose = goal_pose

        while not self.action_client_.wait_for_server(timeout_sec=25.0):
            print('Action client server is not available, restarting gazebo...')
            self.restart_gazebo()

        self._send_goal_future = self.action_client_.send_goal_async(
            goal_msg)
        
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            return

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        status = future.result().status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.path_is_valid = True
        else:
            print("Failed to get path")

        self.result_path = result.path.poses
        self.new_path_received = True

    def get_actions(self):
        """Defines the actions that the environment can have

        # Argument
            None

        # Returns
            The list of possible actions
        """
        raise NotImplementedError()

    def compute_reward(self):

        reward = 0.0
        goal_dist_sq = self.sq_distance_to_goal()

        obstacle_reward = - (1 / (min(self.states_input)**2)) * 0.05

        if goal_dist_sq > 0.25:
            distance_reward = -(goal_dist_sq)
            heading_reward = -0.5 * self.get_heading()**2
        else:
            distance_reward = 1000
            heading_reward = 1000
            self.done = True
            print("Goal Reached")

        elif goal_dist_sq < 0.25:
            reward = 500.0
            self.done = True
            self.hard_reset = True
            print("Goal Reached Reward: {}".format(reward))
            return reward, self.done
        
        elif self.reached_point:
            # reward += 50.0 * (self.checkpoint_index + 1)
            reward = 500.0
            print("checkpoint Reward: {}".format(reward))
            self.done = True
            self.hard_reset = False
            return reward, self.done

        else:          
            linear_velocity_sign = -0.2 if self.linear_velocity < 0 else 0.2
            reward = -1 + linear_velocity_sign
            
            # reward = -self.distance_to_checkpoint[0]*0.1
            #          -self.distance_to_checkpoint[1]*0.2
            #          -self.distance_to_checkpoint[2]*0.3
            #          -self.distance_to_checkpoint[3]*0.4
            #          -self.distance_to_checkpoint[4]*0.5
            #          -0.1
            #          +linear_velocity_sign

            self.done = False
            self.hard_reset = False

        reward += distance_reward
        reward += heading_reward
        reward += obstacle_reward

        if self.collision:
            reward = -500
            self.done = True
        return reward, self.done

    def set_random_robot_pose(self):
        pose = self.get_random_pose()
        pose.position.x = -2.0 #random.uniform(-2, 2)
        pose.position.y = -0.5

        self.set_entity_state_pose('turtlebot3_waffle', pose)

    def set_random_goal_pose(self):
        '''
        Generates a random goal pose, and sets this goal pose in the Gazebo env.
        '''
        self.goal_pose = self.get_random_pose()
        self.set_entity_state_pose('goal_pose', self.goal_pose)

    def get_random_pose(self):
        random_pose = Pose()
        yaw = random.uniform(0, pi * 2)

        random_pose.position.x = random.uniform(-2, 2)
        random_pose.position.y = random.uniform(-2, 2)
        random_pose.position.z = 0.0
        random_pose.orientation.x = 0.0
        random_pose.orientation.y = 0.0
        random_pose.orientation.z = sin(yaw * 0.5)
        random_pose.orientation.w = cos(yaw * 0.5)
        
        return random_pose

    def sq_distance_to_goal(self):
        self.get_robot_pose()
        dx = self.goal_pose.position.x - self.current_pose.position.x
        dy = self.goal_pose.position.y - self.current_pose.position.y
        return dx * dx + dy * dy

    def get_heading(self):
        goal_angle = math.atan2(self.goal_pose.position.y - self.current_pose.position.y,
                                self.goal_pose.position.x - self.current_pose.position.x)

        current_yaw = self.get_yaw(self.current_pose)
        heading = goal_angle - current_yaw

        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        return heading

    def get_yaw(self, q):
        yaw = atan2(2.0 * (q.orientation.x * q.orientation.y + q.orientation.w * q.orientation.z),
                    q.orientation.w * q.orientation.w + q.orientation.x * q.orientation.x -
                    q.orientation.y * q.orientation.y - q.orientation.z * q.orientation.z)
        return yaw

    def observation(self):
        self.get_robot_pose()

        sq_dist = self.sq_distance_to_goal()
        
        states = []
        states.clear()
        for i in range(len(self.laser_scans)):
            states.append(self.laser_scans[i])

        checkpoint = []
        checkpoint.clear()
        for i in range(len(self.checkpoints)):
            checkpoint.append(self.checkpoints[i][0])
            checkpoint.append(self.checkpoints[i][1])

        reached, iteration = self.reached_checkpoint()
        self.reached_point = reached
        self.checkpoint_index = iteration
        if reached == True:
            self.path_index += (iteration+1) * self.path_resolution
            self.get_checkpoints()
        self.publish_checkpoints_marker()

        laser_scans = copy.deepcopy(states)
        current_x = copy.deepcopy([float(self.current_pose.position.x)])
        current_y = copy.deepcopy([float(self.current_pose.position.y)])
        current_yaw = copy.deepcopy([self.get_yaw(self.current_pose)])
        current_goal_x = copy.deepcopy([float(self.goal_pose.position.x)])
        current_goal_y = copy.deepcopy([float(self.goal_pose.position.y)])
        checkpoints_position = copy.deepcopy(checkpoint)
        obs = np.concatenate([
            laser_scans,
            current_x,
            current_y,
            current_yaw,
            current_goal_x,
            current_goal_y,
            checkpoints_position,
        ])

        achieved_goal = copy.deepcopy(np.array([float(self.current_pose.position.x),
                                                float(self.current_pose.position.y)]))
        desired_goal = copy.deepcopy(np.array([float(self.goal_pose.position.x),
                                               float(self.goal_pose.position.y)]))

        # return {
        #     'observation': obs.copy(),
        #     'achieved_goal': achieved_goal.copy(),
        #     'desired_goal': desired_goal.copy(),
        # }
        return obs

    def get_checkpoints(self):
        counter = 0
        self.checkpoints.clear()
        self.checkpoints = [[0.0,0.0]] * self.num_check_points
        for self.local_path_index in range(self.path_index, len(self.path), self.path_resolution):                
            if counter == self.num_check_points:
                break
            else:
                self.checkpoints[counter] = self.path[self.local_path_index]
                counter += 1
        while counter < self.num_check_points:
            if self.local_path_index >= len(self.path):
                self.local_path_index = len(self.path)-1
            self.checkpoints[counter] = self.path[self.local_path_index]
            counter += 1

        return np.array(self.checkpoints)
        
    def reached_checkpoint(self):
        
        for it in range(0, len(self.checkpoints), 1):        
            dx = self.checkpoints[it][0] - self.current_pose.position.x
            dy = self.checkpoints[it][1] - self.current_pose.position.y
            dist = dx * dx + dy * dy
            self.distance_to_checkpoint[it] = dist
            if dist < 0.0625:
                print(it)
                return [True, it]
        return [False, it]

    def publish_checkpoints_marker(self):
        markerArray = MarkerArray()
        for i in range(0, len(self.checkpoints), 1): 
            marker = self.add_marker()
            marker.scale.z = 0.125
            marker.color.g = 0.0
            marker.color.r = 1.0 * ((i+1)/5)
            marker.color.b = 1.0 * ((i+1)/5)
            marker.id = i
            marker.pose.position.x = self.checkpoints[i][0]
            marker.pose.position.y = self.checkpoints[i][1]
            marker.text = "Checkpoint: {}".format(i)
            markerArray.markers.append(marker)

        # Goal marker
        marker = self.add_marker()
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.id = i+1
        marker.pose.position.x = self.goal_pose.position.x
        marker.pose.position.y = self.goal_pose.position.y
        marker.text = 'Goal_pose'
        markerArray.markers.append(marker)

        # Current pose marker
        marker = self.add_marker()
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.id = i+2
        marker.pose.position.x = self.current_pose.position.x
        marker.pose.position.y = self.current_pose.position.y
        marker.text = 'Current_pose'
        markerArray.markers.append(marker)

        self.pub_checkpoints.publish(markerArray)

    def add_marker(self):
        marker = Marker()
        marker.header.frame_id = "/map"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.125
        marker.scale.y = 0.125
        marker.scale.z = 0.0
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        return marker


    def reset(self):
        """
        Resets the turtlebot environment.
        Gets a new goal pose.
        Gets a path to the new goal pose from subscriber.
        """

        if self.hard_reset:
            self.hard_reset = False
            if self.gazebo_started == False:
                self.restart_gazebo()
            
            self.count  = 0
            while not self.path_is_valid:

                self.count += 1
                if self.count > 5:
                    self.restart_gazebo()
                    self.count = 0
                self.count = 0

                self.reset_tb3_env()
                self.set_random_robot_pose()
                self.set_random_goal_pose()
                sleep(1.0)
                self.get_path()
                
            #self.reset_gazebo_world()
            self.initialize_checkpoints()
            self.hard_reset = False
        self.path_fail_count = 0
        self.max_step = 0

        return self.observation()
    
    def initialize_checkpoints(self):
        self.path_is_valid = False
        self.path = []
        for i in range(len(self.result_path)):
             self.path.append([float(self.result_path[i].pose.position.x),
                               float(self.result_path[i].pose.position.y)])

        self.path_resolution = 10
        if len(self.result_path) <= self.path_resolution * self.num_check_points:
            self.path_resolution = int(len(self.result_path) / self.num_check_points)
            if self.path_resolution == 0:
                self.path_resolution = 1
        
        self.checkpoints.clear()
        self.checkpoints = [[0.0,0.0]] * self.num_check_points
        count = 0
        if len(self.path) is not 0:
            for i in range(len(self.path)):
                self.checkpoints.append([self.path[i][0], self.path[i][1]])
                count = i
                if i == 4:
                    break

            if count < self.num_check_points:
                while count < 4:
                    self.checkpoints.append([self.path[i][0], self.path[i][1]])
                    count += 1
        else:
            self.checkpoints = [[0.0,0.0]] * self.num_check_points

        self.path_index = self.path_resolution
        self.get_checkpoints()

        return self.observation()

    def get_path(self):
        '''
        This function will internally calls ComputePathToPose action service,
        and saves the path.
        '''
        goal_msg = PoseStamped()
        
        goal_msg.pose.position.x = self.goal_pose.position.x
        goal_msg.pose.position.y = self.goal_pose.position.y

        self.send_goal(goal_msg)

        while not self.new_path_received:
            sleep(0.1)
            self.stop_action()
        self.new_path_received = False
