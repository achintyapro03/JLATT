import numpy as np
from Robot_w_sens_and_comm import Robot_w_sensors_comm

class Robot_w_dist_JLATT(Robot_w_sensors_comm):
    def __init__(self, x0, name, stat_data, sensor_data, connection_data, debug):
        super().__init__(x0, name, stat_data, sensor_data, connection_data)
        self.R_mes = np.eye(2) * self.v_r
        self.debug = debug
        self.save_est = []
        self.save_cov = []

    @staticmethod
    def track_to_track_fusion(x_set, p_set):
        n = len(x_set)
        alpha_weights = np.zeros(n)
        for i in range(n):
            alpha_weights[i] = 1 / np.trace(p_set[i])
        alpha_weights /= np.sum(alpha_weights)

        p_ci_inv = np.zeros_like(p_set[0])
        for i in range(n):
            p_ci_inv += alpha_weights[i] / p_set[i]
        p_ci = np.linalg.inv(p_ci_inv)

        x_ci = np.zeros_like(x_set[0])
        for i in range(n):
            x_ci += alpha_weights[i] * np.linalg.solve(p_set[i], x_set[i])
        return x_ci, p_ci

    def corrections_from_com(self, t):
        n_corrections = len(self.data_rel)
        s_set = []
        y_set = []

        for l in range(len(self.data_rel)):
            data_row = self.data_rel[l]
            sender_name = data_row['Name']
            sender_zr_row = np.where(self.z_r['Name'] == sender_name)[0]
            zr_row = self.z_r[sender_zr_row]

            z_il = np.array([data_row['Distance'], data_row['Direction']])
            z_li = np.array([zr_row['Distance'], zr_row['Direction']])

            Pose_l = data_row['Pose_estimate'][0]
            p_l = data_row['Error_covariance'][0]
            Pose_i = self.Pose_est
            p_i = self.p_est

            distance_diff = z_il[0] - z_li[0]
            phi_il = (np.angle(z_il[1]) + Pose_l[2]) % (2 * np.pi)
            phi_li = (np.angle(z_li[1]) + Pose_i[2]) % (2 * np.pi)
            dphi_il = (phi_il - phi_li + np.pi) % (2 * np.pi)
            if dphi_il > (2 * np.pi - 0.001):
                dphi_il = 0
            dz_il = np.array([distance_diff, dphi_il])
            dz_il[dz_il < 0.001] = 0

            dx = Pose_l[0] - Pose_i[0]
            dy = Pose_l[1] - Pose_i[1]
            H_i = np.array([[-dx / np.sqrt(dx**2 + dy**2), -dy / np.sqrt(dx**2 + dy**2), 0],
                            [-dy / (dx**2 + dy**2), dx / (dx**2 + dy**2), -1]])
            H_l = np.array([[dx / np.sqrt(dx**2 + dy**2), dy / np.sqrt(dx**2 + dy**2), 0],
                            [dy / (dx**2 + dy**2), -dx / (dx**2 + dy**2), 0]])

            R_hat = self.R_mes + np.dot(H_l, np.dot(p_l, H_l.T))
            R_hat[R_hat < self.v_r] = 0
            R_hat += 0.01 * np.eye(R_hat.shape[0])

            s_l = np.dot(H_i.T, np.linalg.inv(R_hat)).dot(H_i)
            z_corr = np.linalg.inv(R_hat).dot(dz_il + np.dot(H_i, Pose_i))
            z_corr[1] = (z_corr[1] + np.pi) % (2 * np.pi) - np.pi
            y_l = np.dot(H_i.T, z_corr)

            s_set.append(s_l)
            y_set.append(y_l)
        if not s_set:
            pass
        else:
            nu = 1 / n_corrections
            inv_p_hat = sum(nu * np.array(s_set))
            inv_p_hat[inv_p_hat < 0.1] = 0
            inv_p_hat += 1e-3 * np.eye(inv_p_hat.shape[0])
            p_hat = np.linalg.inv(inv_p_hat)

            x_hat = sum(nu * inv_p_hat.dot(np.array(y_set))) #Check this out
            x_hat[2] = (x_hat[2] + np.pi) % (2 * np.pi) - np.pi

            p_est = self.p_est
            p_est[p_est < 0.1] = 0
            p_est += 1e-3 * np.eye(p_est.shape[0])
            Omega = np.linalg.inv(p_est)
            q = Omega.dot(self.Pose_est)

            hat_contr = np.trace(p_hat) ** -1
            est_contr = np.trace(p_est) ** -1
            alpha = hat_contr / (hat_contr + est_contr + 1e-5)
            temp = alpha * inv_p_hat + (1 - alpha) * Omega
            temp += 1e-3 * np.eye(temp.shape[0])
            Gamma = inv_p_hat.dot(np.linalg.inv(temp)).dot(Omega)
            K = Omega - alpha * Gamma
            L = inv_p_hat - (1 - alpha) * Gamma
            new_p_est = np.linalg.inv(Omega + inv_p_hat - Gamma)
            new_Pose = new_p_est.dot(K.dot(np.linalg.inv(Omega)).dot(q) + L.dot(x_hat))
            new_Pose[2] = (new_Pose[2] + 2 * np.pi) % (2 * np.pi)

            if self.debug:
                save = np.concatenate((self.Pose_est, new_Pose, self.true_Pose))
                self.save_est.append(save)
                save = np.concatenate((self.p_est.flatten(), new_p_est.flatten()))
                self.save_cov.append(save)
                e_before = np.abs(self.Pose_est - self.true_Pose)
                e_after = np.abs(new_Pose - self.true_Pose)
                if np.all(e_before < e_after):
                    print(f" - This is making things worse for robot {self.name} at time {t}")
                else:
                    print(f" + Update useful for robot {self.name} at time {t}")
                if e_after[2] > np.pi / 2:
                    print(f" - Robot {self.name} is definitely not going in the right direction at time {t}")
                elif e_before[2] < e_after[2]:
                    print(f" - the update deviating robot {self.name} at time {t}")

            if np.sum(np.abs(new_Pose - self.Pose_est)) > 10:
                print(f" ? Ignored update of robot {self.name} at time {t}")
            else:
                self.p_est = new_p_est
                self.Pose_est = new_Pose
    def simple_corrections(self, t):
        n_corrections = len(self.data_rel)
        s_set = []
        y_set = []
        # print(len(self.data_rel))
        for l in range(len(self.data_rel)):
            print(self.data_rel)
            data_row = self.data_rel.iloc[l]
            sender_name = data_row['Name']
            sender_zr_row = np.where(self.z_r['Name'] == sender_name)[0]
            # print(self.z_r)
            zr_row = self.z_r.iloc[sender_zr_row[0]]

            z_il = np.array([data_row['dX'], data_row['dY']])
            z_li = np.array([zr_row['dX'], zr_row['dY']])
            # print(data_row['Error_covariance'])
            Pose_l = data_row['Pose_estimate'][0]
            p_l = data_row['Error_covariance']
            Pose_i = self.pose_est
            p_i = self.p_est

            dz_il = np.abs(z_il) - np.abs(z_li)

            H_i = np.array([[-1, 0, 0], [0, -1, 0]])
            H_l = np.array([[1, 0, 0], [0, 1, 0]])
            R_hat = self.R_mes + np.dot(H_l, np.dot(p_l, H_l.T))
            R_hat[R_hat < self.v_r] = 0
            R_hat += 0.01 * np.eye(R_hat.shape[0])

            s_l = np.dot(H_i.T, np.linalg.inv(R_hat)).dot(H_i)
            z_corr = np.linalg.inv(R_hat).dot(dz_il + np.dot(H_i, Pose_i))
            z_corr[1] = (z_corr[1] + np.pi) % (2 * np.pi) - np.pi
            y_l = np.dot(H_i.T, z_corr)

            s_set.append(s_l)
            y_set.append(y_l)
        if not s_set:
            pass
        else:
            nu = 1 / n_corrections
            inv_p_hat = sum(nu * np.array(s_set))
            inv_p_hat[inv_p_hat < 0.1] = 0
            inv_p_hat += 1e-3 * np.eye(inv_p_hat.shape[0])
            p_hat = np.linalg.inv(inv_p_hat)
            print(inv_p_hat.shape,np.array(y_set).T.shape)
            x_hat = 0
            # print(inv_p_hat.shape,(np.array(y_set[i]).T).shape)
            for i in range((np.array(y_set).T).shape[0]):
                x_hat = x_hat + (nu * inv_p_hat.dot(np.array(y_set[i]).T))
                # print("Iteration:",i,"x_hat = ",x_hat)
            x_hat[2] = (x_hat[2] + np.pi) % (2 * np.pi) - np.pi

            p_est = self.p_est
            p_est[p_est < 0.1] = 0
            p_est += 1e-3 * np.eye(p_est.shape[0])
            Omega = np.linalg.inv(p_est)
            q = Omega.dot(self.pose_est)

            hat_contr = np.trace(p_hat) ** -1
            est_contr = np.trace(p_est) ** -1
            alpha = hat_contr / (hat_contr + est_contr + 1e-5)
            temp = alpha * inv_p_hat + (1 - alpha) * Omega
            temp += 1e-3 * np.eye(temp.shape[0])
            Gamma = inv_p_hat.dot(np.linalg.inv(temp)).dot(Omega)
            K = Omega - alpha * Gamma
            L = inv_p_hat - (1 - alpha) * Gamma
            new_p_est = np.linalg.inv(Omega + inv_p_hat - Gamma)
            new_L = L.dot(x_hat)
            new_Pose = new_p_est.dot(K.dot(np.linalg.inv(Omega)).dot(q))
            # print(L.shape,x_hat.shape)
            new_Pose = new_Pose + new_L
            # print("New_ Pose =",new_Pose)
            new_Pose[2] = (new_Pose[2] + 2 * np.pi) % (2 * np.pi)
            if self.debug:
                save = np.concatenate((self.Pose_est, new_Pose, self.true_Pose))
                self.save_est.append(save)
                save = np.concatenate((self.p_est.flatten(), new_p_est.flatten()))
                self.save_cov.append(save)
                e_before = np.abs(self.Pose_est - self.true_Pose)
                e_after = np.abs(new_Pose - self.true_Pose)
                if np.all(e_before < e_after):
                    print(f" - This is making things worse for robot {self.name} at time {t}")
                else:
                    print(f" + Update useful for robot {self.name} at time {t}")
                if e_after[2] > np.pi / 2:
                    print(f" - Robot {self.name} is definitely not going in the right direction at time {t}")
                elif e_before[2] < e_after[2]:
                    print(f" - the update deviating robot {self.name} at time {t}")

            if np.sum(np.abs(new_Pose - self.pose_est)) > 10:
                print(f" ? Ignored update of robot {self.name} at time {t}")
            else:
                self.p_est = new_p_est
                self.Pose_est = new_Pose