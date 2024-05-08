import numpy as np
from numpy.linalg import norm
from scipy.stats import wrapcauchy
from Robot import Robot
import matplotlib.pyplot as plt


class RobotWithSensors(Robot):
    def __init__(self, x0, name, stat_data, sensor_data):
        super().__init__(x0, stat_data)
        
        variances, max_distances, probabilities = sensor_data
        
        self.name = name
        
        self.z_abs = []  # absolute measurement
        self.v_abs = variances[0]  # absolute measurement variance
        self.abs_mes_prob = probabilities[0]
        
        self.z_r = []  # distance and direction of other robots
        self.v_r = variances[1]  # robot measurement variance
        self.z_r_max = max_distances[0]  # max measurement distance
        self.rel_mes_prob = probabilities[1]
        
        self.z_t = []  # distance and direction of targets
        self.v_t = variances[2]  # target measurement variance
        self.z_t_max = max_distances[1]  # max measurement distance

    def update_abs_meas(self, time_stamp):
        measured_pose = self.true_pose + self.v_abs * np.random.randn(3)
        if np.random.rand() < self.abs_mes_prob:
            self.z_abs.append({'Name': self.name, 'X': measured_pose[0], 'Y': measured_pose[1], 'Heading': measured_pose[2], 'Time': time_stamp})

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
        self.z_r = []
        for current_robot in robot_set:
            measure = self.rel_sensor_model(current_robot)
            if current_robot.name == self.name:
                continue
            elif measure[0] < self.z_r_max:
                if np.random.rand() < self.rel_mes_prob:
                    self.z_r.append({'Name': current_robot.name, 'Distance': measure[0], 'Direction': measure[1], 'Time': time_stamp})

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
    robot_with_sensors = RobotWithSensors(x0, 'Robot1', stat_data, sensor_data)

    # Time step
    dt = 1  # seconds

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
