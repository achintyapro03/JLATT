import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig, inv

class Robot:
    def __init__(self, x0, stat_data):
        self.true_pose = np.array(x0, dtype=np.float64)
        self.true_pose[2] = self.wrap_to_2pi(self.true_pose[2])
        self.true_path = [self.true_pose.copy()]
        # self.true_path = np.transpose(self.true_pose.copy())

        e_var0, w_var = stat_data

        self.pose_est = np.array(x0, dtype=np.float64)
        self.pose_est[2] = self.wrap_to_2pi(self.pose_est[2])

        self.p_est = np.eye(3) * e_var0
        self.process_cov = np.eye(2) * w_var

    def wrap_to_2pi(self, angle):
        return angle % (2 * np.pi)

    def dynamics(self, dt, v, w):
        # Update the robot's position using the dynamics rule
        phi = self.true_pose[2]
        vel = np.array([v * np.cos(phi), v * np.sin(phi), w])
        self.true_pose += vel * dt
        self.true_pose[2] = self.wrap_to_2pi(self.true_pose[2])
        self.true_path.append(self.true_pose.copy())


    def dynamics_est(self, pose_est, u, dt):
        # Simulate dynamics with process noise
        w = np.random.randn(2) * np.diag(self.process_cov)
        x_next = pose_est + np.array([
            (u[0] - w[0]) * np.cos(pose_est[2]),
            (u[0] - w[0]) * np.sin(pose_est[2]),
            u[1] - w[1]
        ]) * dt
        return x_next

    def propagate(self, dt, v, w):
        pose_est = self.pose_est

##TODO: implement IMPORTANT MATLAB it is row vector
        u = np.array([v, w])

        Phi = np.array([
            [1, 0, -u[0] * dt * np.sin(pose_est[2])],
            [0, 1, u[0] * dt * np.cos(pose_est[2])],
            [0, 0, 1]
        ])
        G = np.array([
            [np.cos(pose_est[2]), 0],
            [np.sin(pose_est[2]), 0],
            [0, 1]
        ])
        Q = G.dot(self.process_cov).dot(G.T)

        self.p_est = Phi.dot(self.p_est).dot(Phi.T) + Q
        self.pose_est = self.dynamics_est(pose_est, u, dt)
        self.pose_est[2] = self.wrap_to_2pi(self.pose_est[2])

    def update(self, dt):
        # Generate random velocities for random movement
        v = 0.5  # m/s
        w = np.random.uniform(-np.pi/4, np.pi/4)  # random angular velocity
        self.dynamics(dt, v, w)
        self.propagate(dt, v, w)

    def draw(self, i):
        # Draw a triangle at a specific position from the path
        x, y, th = self.true_path[i]
        vertices_x = np.array([0, 1, 0]) * 0.1
        vertices_y = np.array([-0.5, 0, 0.5]) * 0.1


        R = np.array([
            [np.cos(th), -np.sin(th)],
            [np.sin(th), np.cos(th)]
        ])
        rotated_vertices = R.dot(np.array([vertices_x, vertices_y]))
        translated_vertices_x = rotated_vertices[0, :] + x
        translated_vertices_y = rotated_vertices[1, :] + y

        return [translated_vertices_x, translated_vertices_y]

    def draw_estimate(self):
        # Plot the estimated pose and uncertainty ellipse
        pose = self.pose_est
        P = self.p_est
        
        plt.plot(pose[0], pose[1], 'bo')  # Plot a blue circle at the estimated position

        # Calculate the uncertainty ellipse for x and y
        P_position = P[:2, :2]
        eigval, eigvec = eig(P_position)
        
        largest_eigenval = max(eigval)
        largest_eigenvec = eigvec[:, np.argmax(eigval)]
        
        smallest_eigenval = min(eigval)
        angle = np.arctan2(largest_eigenvec[1], largest_eigenvec[0])

        # Get the 95% confidence interval error ellipse
        chisquare_val = 5.991  # 95% confidence interval for 2 degrees of freedom
        theta_grid = np.linspace(0, 2 * np.pi, 100)
        phi = angle
        X0, Y0 = pose[:2]
        a = np.sqrt(chisquare_val * largest_eigenval)
        b = np.sqrt(chisquare_val * smallest_eigenval)

        # Create the ellipse in x and y coordinates
        ellipse_x_r = a * np.cos(theta_grid)
        ellipse_y_r = b * np.sin(theta_grid)

        # Rotate the ellipse
        R = np.array([
            [np.cos(phi), np.sin(phi)],
            [-np.sin(phi), np.cos(phi)]
        ])
        r_ellipse = np.dot(np.array([ellipse_x_r, ellipse_y_r]).T, R)

        # Plot the error ellipse
        plt.plot(r_ellipse[:, 0] + X0, r_ellipse[:, 1] + Y0, 'r')

        # Plot the heading
        head_length = 0.5  # Length of the heading line, adjust as needed
        heading_vector = np.array([np.cos(pose[2]), np.sin(pose[2])]) * head_length
        plt.quiver(pose[0], pose[1], heading_vector[0], heading_vector[1], angles='xy', scale_units='xy', scale=1, color='k', width=0.01)

# # Testbench to test the Robot class
# def test_robot():
#     # Initial pose: [x, y, theta]
#     x0 = [0, 0, 0]

#     # Statistics data: [e_var0, w_var]
#     stat_data = [0.01, 0.001]

#     # Create the robot
#     robot = Robot(x0, stat_data)

#     # Time step
#     dt = 0.1  # seconds

#     # Simulate the robot for 100 steps
#     for _ in range(100):
#         robot.update(dt)

#     # Plot the true path and the estimated path
#     true_path = np.array(robot.true_path)
#     plt.plot(true_path[:, 0], true_path[:, 1], 'g-', label='True Path')

#     # Draw the estimated pose and uncertainty ellipse
#     robot.draw_estimate()

#     # Plot the robot triangle at the current position
#     triangle = robot.draw(-1)
#     plt.fill(triangle[0], triangle[1], 'b', alpha=0.5, label='Robot')

#     # Set plot settings
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('Robot Path and Estimated Pose')
#     plt.legend()
#     plt.grid(True)
#     plt.axis('equal')

#     # Show the plot
#     plt.show()

# # Run the testbench
# test_robot()

def test_robot():
    # Initial pose: [x, y, theta]
    x0 = [0, 0, 0]

    # Statistics data: [e_var0, w_var] - Adjusted process noise variance
    stat_data = [0.001, 0.0001]  # Reduced values for process noise

    # Create the robot
    robot = Robot(x0, stat_data)

    # Time step
    dt = 1  # Reduced time step

    # Create a figure
    plt.figure()

    # Simulate the robot for 200 steps (you can adjust the number of steps)
    for step in range(200):
        robot.update(dt)
        # Plot the estimated pose and uncertainty ellipse at each step
        robot.draw_estimate()
        
        # Optionally, add a pause to visualize each step
        plt.pause(0.05)  # Pause to visualize each step

    # Plot the true path
    # true_path = np.array(robot.true_path)
    # plt.plot(true_path[:, 0], true_path[:, 1], 'g-', label='True Path')

    # # Draw the robot triangle at the current position
    # triangle = robot.draw(-1)
    # plt.fill(triangle[0], triangle[1], 'b', alpha=0.5, label='Robot')

    # Set plot settings
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Robot Path and Estimated Pose')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    # Show the plot
    plt.show()

# Run the testbench

if(__name__ == '__main__'):
    test_robot()

