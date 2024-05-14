import numpy as np
from numpy.linalg import norm
from Robot import Robot
import matplotlib.pyplot as plt
import pandas as pd
from numpy.random import rand, randn


class Robot_w_sensors(Robot):
    def __init__(self, x0, name, stat_data, sensor_data):
        super().__init__(x0, stat_data)
        
        variances, max_distances, probabilities = sensor_data

        self.v_abs = variances[0]  # absolute measurement variance
        self.v_r = variances[1]  # robot measurement variance
        self.v_t = variances[2]  # target measurement variance
        
        self.name = name
        
        self.z_abs = pd.DataFrame(columns=['Name', 'X', 'Y', 'Heading', 'Time'])
        self.abs_mes_prob = probabilities[0]
        
        self.z_r = pd.DataFrame(columns=['Name', 'Distance', 'Direction', 'Time'])
        self.z_t = pd.DataFrame(columns=['Name', 'Distance', 'Direction', 'Time'])
        self.z_r_max = max_distances[0]  # max measurement distance
        
        self.rel_mes_prob = probabilities[1]
        # self.z_t_max = max_distances[1]  # max measurement distance
    

    def update_abs_meas(self, time_stamp):
        self.z_abs = pd.DataFrame(columns=['Name', 'X', 'Y', 'Heading', 'Time'])

        measured_pose = self.true_pose + self.v_abs * np.random.randn(3)
        if np.random.rand() < self.abs_mes_prob:
            self.z_abs = pd.DataFrame([[self.name, measured_pose[0], measured_pose[1], measured_pose[2], time_stamp]],
                                      columns=['Name', 'X', 'Y', 'Heading', 'Time'])

    def wrap_to_2pi(self, angle):
        return angle % (2 * np.pi)
    
    def rel_sensor_model(self, robot):
        x_i, y_i, th_i = self.true_pose
        x_l, y_l, th_l = robot.true_pose
        
        dx = x_l - x_i
        dy = y_l - y_i
        
        dist = norm([dx, dy]) + self.v_r * np.random.randn()
        angle = self.wrap_to_2pi(np.arctan2(dy, dx)) - th_i + self.v_r * np.random.randn()
        
        return dist, self.wrap_to_2pi(angle)

    def update_rel_meas(self, robot_set, time_stamp):
        self.z_r = pd.DataFrame(columns=['Name', 'Distance', 'Direction', 'Time'])
        for current_robot in robot_set:
            measure = self.rel_sensor_model(current_robot)
            if current_robot.name == self.name:
                continue
            elif measure[0] < self.z_r_max:
                if np.random.rand() < self.rel_mes_prob:
                    newRow = pd.DataFrame([[current_robot.name, measure[0], measure[1], time_stamp]],
                                        columns=['Name', 'Distance', 'Direction', 'Time'])
                    self.z_r = pd.concat([self.z_r, newRow], ignore_index=True)


    def simple_rel_meas(self, robot_set, time_stamp):
        self.z_r = pd.DataFrame(columns=['Name', 'Distance', 'dX', 'dY', 'Time'])

        for current_robot in robot_set:
            measure = self.rel_sensor_model(current_robot)

            if current_robot.name == self.name:
                continue
            elif measure[0] < self.z_r_max:
                if rand() < self.rel_mes_prob:
                    dX = current_robot.true_pose - self.true_pose
                    newRow = pd.DataFrame([[current_robot.name, measure[0], dX[0], dX[1], time_stamp]],
                                          columns=['Name', 'Distance', 'dX', 'dY', 'Time'])
                    self.z_r = pd.concat([self.z_r, newRow], ignore_index=True)
                    # self.z_r = self.z_r.append(newRow, ignore_index=True)

    def update_meas(self, robot_set, time_stamp):
        self.update_abs_meas(time_stamp)
        self.update_rel_meas(robot_set, time_stamp)



def test_robot_with_sensors():
    # Initial pose: [x, y, theta]
    x0 = [0, 0, 0]
    
    # Statistics data: [e_var0, w_var]
    stat_data = [0.01, 0.001]
    
    # Sensor data: [[v_abs, v_r, v_t], [z_r_max, z_t_max], [abs_mes_prob, rel_mes_prob]]
    sensor_data = [[0.01, 0.1, 0.1], [5, 10], [0.9, 0.9]]

    # Create the robot with sensors
    robot_with_sensors = Robot_w_sensors(x0, 'Robot1', stat_data, sensor_data)

    # Time step
    dt = 0.1  # seconds

    # Create a figure
    plt.figure()

    # Simulate the robot for 200 steps
    for step in range(200):
        # Update the robot's measurements
        robot_with_sensors.update_meas([], step)
        
        # Plot the estimated pose and uncertainty ellipse at each step
        robot_with_sensors.draw_estimate()
        
        # Optionally, add a pause to visualize each step
        plt.pause(0.05)

    # Plot the true path
    true_path = np.array(robot_with_sensors.true_path)
    plt.plot(true_path[:, 0], true_path[:, 1], 'g-', label='True Path')

    # Draw the robot triangle at the current position
    triangle = robot_with_sensors.draw(-1)
    plt.fill(triangle[0], triangle[1], 'b', alpha=0.5, label='Robot')

    # Set plot settings
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Robot Path and Estimated Pose with Sensors')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    # Show the plot
    plt.show()

# Run the testbench

if(__name__ == '__main__'):
    test_robot_with_sensors()



    """
import numpy as np
import pandas as pd
from math import sqrt, atan2
from numpy.random import rand, randn
from numpy import wrap


class Robot_w_sensors(Robot):
    def __init__(self, x0, name, stat_data, sensor_data):
        super().__init__(x0, stat_data)
        variances = sensor_data[0]
        max_distances = sensor_data[1]
        probabilities = sensor_data[2]

        self.v_abs = variances[0]
        self.v_r = variances[1]
        self.v_t = variances[2]

        self.name = name

        self.z_abs = pd.DataFrame(columns=['Name', 'X', 'Y', 'Heading', 'Time'])
        self.abs_mes_prob = probabilities[0]

        self.z_r = pd.DataFrame(columns=['Name', 'Distance', 'Direction', 'Time'])
        self.z_t = pd.DataFrame(columns=['Name', 'Distance', 'Direction', 'Time'])
        self.z_r_max = max_distances[0]
        self.rel_mes_prob = probabilities[1]

    def update_abs_meas(self, time_stamp):
        measured_pose = self.true_Pose + self.v_abs * randn(3)
        self.z_abs = pd.DataFrame(columns=['Name', 'X', 'Y', 'Heading', 'Time'])

        if rand() < self.abs_mes_prob:
            self.z_abs = pd.DataFrame([[self.name, measured_pose[0], measured_pose[1], measured_pose[2], time_stamp]],
                                      columns=['Name', 'X', 'Y', 'Heading', 'Time'])

    def rel_sensor_model(self, robot):
        x_i, y_i, th_i = self.true_Pose
        x_l, y_l, th_l = robot.true_Pose

        dx = x_l - x_i
        dy = y_l - y_i

        dist = sqrt(dx**2 + dy**2) + self.v_r * randn()
        angle = wrap(atan2(dy, dx) - th_i + self.v_r * randn(), 0, 2 * np.pi)

        return dist, angle

    def update_rel_meas(self, robot_set, time_stamp):
        self.z_r = pd.DataFrame(columns=['Name', 'Distance', 'Direction', 'Time'])

        for current_robot in robot_set:
            measure = self.rel_sensor_model(current_robot)

            if current_robot.name == self.name:
                continue
            elif measure[0] < self.z_r_max:
                if rand() < self.rel_mes_prob:
                    newRow = pd.DataFrame([[current_robot.name, measure[0], measure[1], time_stamp]],
                                          columns=['Name', 'Distance', 'Direction', 'Time'])
                    self.z_r = self.z_r.append(newRow, ignore_index=True)

    def simple_rel_meas(self, robot_set, time_stamp):
        self.z_r = pd.DataFrame(columns=['Name', 'Distance', 'dX', 'dY', 'Time'])

        for current_robot in robot_set:
            measure = self.rel_sensor_model(current_robot)

            if current_robot.name == self.name:
                continue
            elif measure[0] < self.z_r_max:
                if rand() < self.rel_mes_prob:
                    dX = current_robot.true_Pose - self.true_Pose
                    newRow = pd.DataFrame([[current_robot.name, measure[0], dX[0], dX[1], time_stamp]],
                                          columns=['Name', 'Distance', 'dX', 'dY', 'Time'])
                    self.z_r = self.z_r.append(newRow, ignore_index=True)

    def update_meas(self, robot_set, time_stamp):
        self.update_abs_meas(time_stamp)
        self.update_rel_meas(robot_set, time_stamp)

 """
