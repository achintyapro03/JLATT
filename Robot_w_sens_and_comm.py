import numpy as np
from numpy.linalg import norm
from scipy.stats import wrapcauchy
from Robot import Robot
import matplotlib.pyplot as plt
from Robot_w_sensors import Robot_w_sensors
import pandas as pd

class Robot_w_sensors_comm(Robot_w_sensors):
    def __init__(self, x0, name, stat_data, sensor_data, connection_data):
        super().__init__(x0, name, stat_data, sensor_data)
        
        link_max_dist, link_prob = connection_data
        
        self.link_max_dist = link_max_dist
        self.abs_link_prob = link_prob[0]
        self.rel_link_prob = link_prob[1]
        
        self.data_rel = pd.DataFrame(columns=['Name', 'Distance','Direction','Pose_estimate', 'Error_covariance','Time'])


    def update_rel_com(self, robot_set, time_stamp):
        self.data_rel = pd.DataFrame(columns=['Name', 'Distance','Direction','Pose_estimate', 'Error_covariance','Time'])
        
        for robot in robot_set:
            his_row = np.where(self.z_r['Name'] == robot.name)[0]
            my_row = np.where(robot.z_r['Name'] == self.name)[0]
            
            if len(his_row) == 0 or len(my_row) == 0:
                continue
            
            distance = self.z_r['Distance'][his_row[0]]
            my_distance = robot.z_r['Distance'][my_row[0]]
            my_direction = robot.z_r['Direction'][my_row[0]]
            
            if distance < self.link_max_dist and my_distance < self.link_max_dist:
                if np.random.rand() < self.rel_link_prob:
                    rel_data = {'Name': robot.name, 'Distance': my_distance, 'Direction': my_direction, 'Pose_estimate': robot.pose_est, 'Error_covariance': robot.p_est, 'Time': time_stamp}
                    # self.data_rel.append(rel_data)
                    self.data_rel = pd.concat([self.data_rel, pd.DataFrame(rel_data)], ignore_index=True)

    def simple_rel_com(self, robot_set, time_stamp):
        self.data_rel = pd.DataFrame(columns=['Name', 'dX','dY','Pose_estimate', 'Error_covariance','Time'])
        
        for robot in robot_set:
            his_row = np.where(self.z_r['Name'] == robot.name)[0]
            my_row = np.where(robot.z_r['Name'] == self.name)[0]
            
            if len(his_row) == 0 or len(my_row) == 0:
                continue
            
            distance = self.z_r['Distance'][his_row[0]]
            my_distance = robot.z_r['Distance'][my_row[0]]
            my_dx = robot.z_r['dX'][my_row[0]]
            my_dy = robot.z_r['dY'][my_row[0]]
            
            if distance < self.link_max_dist and my_distance < self.link_max_dist:
                if np.random.rand() < self.rel_link_prob:
                    rel_data = pd.DataFrame([[robot.name, my_dx, my_dy, robot.pose_est,robot.p_est, time_stamp]],
                                          columns=['Name','dX', 'dY','Pose_estimate','Error_covariance', 'Time'])
                    self.data_rel = pd.concat([self.data_rel, rel_data], ignore_index=True)


    def update_com(self, robot_set, time_stamp):
        self.update_rel_com(robot_set, time_stamp)
