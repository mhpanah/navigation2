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

from core_env import Env

from time import sleep
from threading import Thread
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
#import tf2_ros

from std_srvs.srv import Empty
from gazebo_msgs.srv import GetEntityState, SetEntityState

import os
import subprocess
import signal


class GazeboInterface(Env):
    def __init__(self):
        super()
        self.GazeboInterface = GazeboInterface
        self.node_ = rclpy.create_node('gazebo_interface')
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node_)
        self.node_.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.reset_simulation = self.node_.create_client(Empty, 'reset_simulation')
        self.reset_world = self.node_.create_client(Empty, 'reset_world')
        self.unpause_proxy = self.node_.create_client(Empty, 'unpause_physics')
        self.pause_proxy = self.node_.create_client(Empty, 'pause_physics')
        self.get_entity_state = self.node_.create_client(GetEntityState, 'get_entity_state')
        self.set_entity_state = self.node_.create_client(SetEntityState, 'set_entity_state')
        self.t = Thread(target=self.executor.spin)
        self.t.start()
        self.time_factor = 1.0
        self.time_to_sample = 1.0
        self.count = 0
        nav2_system_tests_dir = get_package_share_directory('nav2_system_tests')
        self.world_model_path=os.path.join(nav2_system_tests_dir, 'models/room1/world.model')
        self.gazebo_started = False
        self.gazebo_process = subprocess.Popen(['gazebo', '-s', 'libgazebo_ros_init.so', 
                                                 self.world_model_path])
        self.gazebo_started = True
        #self.tf_broadcaster = TransformBroadcaster(self.node_)
        #self.send_transform()
        
    def get_robot_pose(self):
        """Gets the robot pose with respect to the world
        Argument:
            None
        Returns:
            The robot pose
        """
        raise NotImplementedError()

    def cleanup(self):
        self.t.join()

    # Rate object is not yet available in rclpy. Thus, we created this method to calculate the
    # difference between simulation time and system time
    def get_time_factor(self):
        sim_time_start = self.node_._clock.now()
        sleep(self.time_to_sample)
        sim_time_end = self.node_._clock.now()
        sim_time_dif = (sim_time_end.nanoseconds - sim_time_start.nanoseconds) / 1e9
        return sim_time_dif / self.time_to_sample

    def set_entity_state_pose(self, entity_name, entity_pose):
        while not self.set_entity_state.wait_for_service(timeout_sec=1.0):
            print('Set entity state service is not available...')

        req = SetEntityState.Request()
        req.state.name = entity_name
        req.state.pose.position = entity_pose.position
        req.state.pose.position.z = 0.0
        req.state.pose.orientation = entity_pose.orientation
        future = self.set_entity_state.call_async(req)
        #rclpy.spin_until_future_complete(self.node_, future)
        while not future.done() and rclpy.ok():
            sleep(0.01)
        sleep(0.5)

    def pause_gazebo_world(self):
        while not self.pause_proxy.wait_for_service(timeout_sec=0.1):
            print('Pause Environment service is not available...')
            self.count += 1
            if self.count > 5:
                self.restart_gazebo()
                self.count = 0
        self.count = 0
        self.pause_proxy.call_async(Empty.Request())

    def unpause_gazebo_world(self):
        while not self.unpause_proxy.wait_for_service(timeout_sec=0.1):
            print('Unpause Environment service is not available...')
            self.count += 1
            if self.count > 5:
                self.restart_gazebo()
                self.count = 0
        self.count = 0
        self.unpause_proxy.call_async(Empty.Request())

    def reset_gazebo_world(self):
        while not self.reset_world.wait_for_service(timeout_sec=1.0):
            print('Reset world service is not available...')
            self.count += 1
            if self.count > 5:
                self.restart_gazebo()
                self.count = 0
        self.count = 0
        self.reset_world.call_async(Empty.Request())

    def reset_gazebo_simulation(self):
        while not self.reset_simulation.wait_for_service(timeout_sec=1.0):
            print('Reset simulation service is not available...')
            self.count += 1
            if self.count > 5:
                self.restart_gazebo()
                self.count = 0
        self.count = 0
        self.reset_simulation.call_async(Empty.Request())

    def restart_gazebo(self):
        self.kill_gazebo()
        self.gazebo_process = subprocess.Popen(['gazebo', '-s', 'libgazebo_ros_init.so',
                                                 self.world_model_path])
        print(self.gazebo_process.pid)
        #self.send_transform()
        self.gazebo_started == True
    
    def kill_gazebo(self):
        os.kill(self.gazebo_process.pid, signal.SIGKILL)
        self.gazebo_started == False

#     def send_transform(self):
#         # Fill up the static transform message
#         static_transformStamped = TransformStamped()
#         
#         current_time = self.node_._clock.now()

#         static_transformStamped.header.stamp = current_time.to_msg()
#         static_transformStamped.header.frame_id = "map" #odom
#         static_transformStamped.child_frame_id = "odom" #base_link

#         static_transformStamped.transform.translation.x = 0.0 # initial_robot_x
#         static_transformStamped.transform.translation.y = 0.0 # initial_robot_y
#         static_transformStamped.transform.translation.z = 0.0

#         static_transformStamped.transform.rotation.x = 0.0 # initial_robot_...
#         static_transformStamped.transform.rotation.y = 0.0
#         static_transformStamped.transform.rotation.z = 0.0
#         static_transformStamped.transform.rotation.w = 1.0

#         self.tf_broadcaster.sendTransform(static_transformStamped)
#         print("transform sent")