import random


import numpy as np
from math import *

import matplotlib.pyplot as plt
from matplotlib import animation

WORLD_SIZE_X = 25   #가로
WORLD_SIZE_Y = 15   #세로

LANDMARKS = [[5.5, 11.5],    # 왼쪽 위
            [16.5, 9.5],     # 오른쪽 위
            [5.5, 4.5],      # 왼쪽 아래
            [15.5, 2.5]]     # 오른쪽 아래

LANDMARKS = np.array(LANDMARKS)

N_LANDMARKS = len(LANDMARKS)

omega = np.array([[[0.0, 0.0] for _ in range(N_LANDMARKS + 1)]

                              for _ in range(N_LANDMARKS + 1)])

xi = np.array([[0.0, 0.0] for _ in range(N_LANDMARKS + 1)])

omega[0, 0, :] = np.array([1.0, 1.0])

xi[0, :] = np.array([22.5, 12.5])

HANDLE_NOISE_STD = 0.1
MEASUREMENT_NOISE_STD = 0.5
MOTION_NOISE_STD = 0.1




# ------------------------------------------------

#

# this is the Robot class

#


class Robot(object):

    def __init__(self, length=1):

        """

        Creates robot and initializes location/orientation to 0, 0, 0.

        """

        self.x = 0.0

        self.y = 0.0

        self.orientation = 0

        self.length = length

        self.steering_noise = 0.0

        self.distance_noise = 0.0

        self.steering_drift = 0.0

        self.measurement_noise = 0.0


    def set(self, x, y, orientation):

        """

        Sets a robot coordinate.

        """

        self.x = x

        self.y = y

        self.orientation = orientation % (2.0 * np.pi)

    def set_noise(self, steering_noise, distance_noise, measurement_noise):

        """

        Sets the noise parameters.

        """

        # makes it possible to change the noise parameters

        # this is often useful in particle filters

        self.steering_noise = steering_noise

        self.distance_noise = distance_noise

        self.measurement_noise = measurement_noise

    def set_steering_drift(self, drift):

        """

        Sets the systematical steering drift parameter

        """

        self.steering_drift = drift

    def move(self, steering, distance, tolerance=0.001, max_steering_angle=np.pi/3): #lab11의 모델 포함

        """

        steering = front wheel steering angle, limited by max_steering_angle

        distance = total distance driven, most be non-negative

        """

        # apply noise
        # print(steering)
        # print(self.orientation)
        steering2 = steering - self.orientation + random.gauss(0.0, self.steering_noise) # steering = 알파 (바이시클 모델)

        steering2 = normalize_angle(steering2)

        # print(steering2)
        #핸들의 허용 각도가 넘어가면 최대 각도로 설정
        if steering2 > max_steering_angle:

            steering2 = max_steering_angle

        if steering2 < -max_steering_angle:

            steering2 = -max_steering_angle

        distance2 = random.gauss(distance, self.distance_noise) # distance = d (바이시클 모델)

        # Execute motion

        turn = np.tan(steering2) * distance2 / self.length # turn = 베타(바이시클모델)

        if abs(turn) < tolerance:  # 움직임이 미세할 경우 바이시클 모델이 아닌 점이 움직이는 걸로 가정하고 계산

            # approximate by straight line motion

            self.x += distance2 * np.cos(self.orientation)

            self.y += distance2 * np.sin(self.orientation)

            self.orientation = (self.orientation + turn) % (2.0 * np.pi)

        else:    # 움직임이 적당히 클 경우

            # approximate bicycle model for motion

            radius = distance2 / turn # = 회전 반경

            # 회전 중심 위치
            cx = self.x - (np.sin(self.orientation) * radius)

            cy = self.y + (np.cos(self.orientation) * radius)

            # 다음 상태의 위치 + 자세
            self.orientation = (self.orientation + turn) % (2.0 * np.pi) # = 세타 + 베타

            self.x = cx + (np.sin(self.orientation) * radius)

            self.y = cy - (np.cos(self.orientation) * radius)

    def sense(self):

        Z = []

        for i, landmark in enumerate(LANDMARKS):

            dx = landmark[0] - self.x + random.uniform(-1, 1)*self.measurement_noise

            dy = landmark[1] - self.y + random.uniform(-1, 1)*self.measurement_noise

            Z.append([i, dx, dy])

        return Z

    def __repr__(self):

        return '[x=%.5f y=%.5f orient=%.5f]' % (self.x, self.y, self.orientation)

    # motion noise가 없는 경우의 움직임
    def ideal_move(self, steering, distance, tolerance=0.001, max_steering_angle=np.pi/3):
        steering2 = steering - self.orientation # steering = 알파 (바이시클 모델)

        steering2 = normalize_angle(steering2)

        #핸들의 허용 각도가 넘어가면 최대 각도로 설정
        if steering2 > 0 and  2 * np.pi - steering2  > max_steering_angle:

            steering2 = max_steering_angle

        if steering2 < 0 and steering2 < -max_steering_angle:

            steering2 = -max_steering_angle


        distance2 = distance # distance = d (바이시클 모델)

        # Execute motion

        turn = np.tan(steering2) * distance2 / self.length # turn = 베타(바이시클모델)

        if abs(turn) < tolerance:  # 움직임이 미세할 경우 바이시클 모델이 아닌 점이 움직이는 걸로 가정하고 계산'

            # approximate by straight line motion

            dx = distance2 * np.cos(self.orientation)

            dy = distance2 * np.sin(self.orientation)

            return dx, dy

        else:    # 움직임이 적당히 클 경우

            # approximate bicycle model for motion

            radius = distance2 / turn # = 회전 반경

            # 회전 중심 위치
            cx = self.x - (np.sin(self.orientation) * radius)

            cy = self.y + (np.cos(self.orientation) * radius)

            # 다음 상태의 위치 + 자세
            orientation = (self.orientation + turn) % (2.0 * np.pi) # = 세타 + 베타

            dx =  (np.sin(orientation) * radius) - (np.sin(self.orientation) * radius)

            dy =   (np.cos(self.orientation) * radius) - (np.cos(orientation) * radius)

            return dx, dy
        
def normalize_angle(angle):
    """Normalize the angle to be within the range [-pi, pi]."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

# slam
def slam(i, dx, dy, Z):

    global omega, xi

    measurement_noise = random.gauss(0.0, MEASUREMENT_NOISE_STD)

    motion_noise = random.gauss(0.0, MOTION_NOISE_STD)

    omega = np.insert(omega, i + 1, 0, axis=0)

    omega = np.insert(omega, i + 1, 0, axis=1)

    xi = np.insert(xi, i + 1, 0, axis=0)

    epsilon = 1e-10

    omega[i + 1, i + 1] += epsilon

    for meas in Z:

        j, x, y = meas

        omega[i, i] = omega[i, i] + 1/measurement_noise

        omega[i, i + j + 2] = omega[i, i + j + 2] - 1/measurement_noise

        omega[i + j + 2, i] = omega[i + j + 2, i] - 1/measurement_noise

        omega[i + j + 2, i + j + 2] = omega[i + j + 2, i + j + 2] + 1/measurement_noise

        xi[i, :] = xi[i, :] - np.array([x, y])/measurement_noise

        xi[i + j + 2, :] = xi[i + j + 2, :] + np.array([x, y])/measurement_noise

    omega[i, i] = omega[i, i] + 1/motion_noise

    omega[i + 1, i + 1] = omega[i + 1, i + 1] + 1/motion_noise

    omega[i + 1, i] = omega[i + 1, i] - 1/motion_noise

    omega[i, i + 1] = omega[i, i + 1] - 1/motion_noise

    xi[i, :] = xi[i, :] - np.array([dx, dy])/motion_noise

    xi[i + 1, :] = xi[i + 1, :] + np.array([dx, dy])/motion_noise

    try:

        mu_x = np.linalg.inv(omega[:, :, 0]).dot(xi[:, 0])

        mu_y = np.linalg.inv(omega[:, :, 1]).dot(xi[:, 1])

    except np.linalg.LinAlgError:

        mu_x = np.linalg.pinv(omega[:, :, 0]).dot(xi[:, 0])

        mu_y = np.linalg.pinv(omega[:, :, 1]).dot(xi[:, 1])

    return np.c_[mu_x, mu_y]


fig, ax = plt.subplots()

robot = Robot()

robot.set(22.5, 12.5, 1.5 * np.pi)

robot.set_noise(HANDLE_NOISE_STD, MOTION_NOISE_STD, MEASUREMENT_NOISE_STD)

ax.plot(LANDMARKS[:, 0], LANDMARKS[:, 1], 'g*', label='Actual Landmark')

estimated_landmarks, = ax.plot([], [], 'k*', label='Estimated Landmark')

actual_position, = ax.plot([], [], 'r.', label='Actual Position')

estimated_position, = ax.plot([], [], 'b.', label='Estimated Position')

actual_path, = ax.plot([], [], 'r--')

estimated_path, = ax.plot([], [], 'b:')

actual_values = []

estimated_values = []

# 실제 로봇과 추정 로봇의 방향 표시 화살표 추가
actual_direction = ax.quiver([], [], [], [], color='r', scale=10)

def init():
    ax.set_xlim(0, WORLD_SIZE_X)
    ax.set_ylim(0, WORLD_SIZE_Y)
    plt.legend(loc='upper left')
    return (actual_position, estimated_position, estimated_landmarks,
            actual_path, estimated_path, actual_direction)

def animate(i):

    print("i: ", i)
    
    handle = ((-2 * i * np.pi / 50) + 1.5 * np.pi)  # 2π로 모듈로 연산

    Z = robot.sense()

    dx, dy = robot.ideal_move(handle, 0.5)

    mu = slam(i, dx, dy, Z)

    robot.move(handle, 0.5)

    actual_values.append([robot.x, robot.y])

    estimated_values.append([mu[i + 1, 0], mu[i + 1, 1]])

    # 경로 업데이트
    actual_path.set_data([pos[0] for pos in actual_values], [pos[1] for pos in actual_values])

    estimated_path.set_data([pos[0] for pos in estimated_values], [pos[1] for pos in estimated_values])

    # 현재 위치 업데이트 (리스트로 변경)
    actual_position.set_data([robot.x], [robot.y])  # 단일 값을 리스트로 변환

    estimated_position.set_data([mu[i + 1, 0]], [mu[i + 1, 1]])  # 단일 값을 리스트로 변환

    # 랜드마크 위치 업데이트
    est_lm = np.array([[mu[i + j + 2, 0], mu[i + j + 2, 1]] for j in range(N_LANDMARKS)])

    estimated_landmarks.set_data(est_lm[:, 0], est_lm[:, 1])

    # 실제 로봇의 방향 화살표 업데이트
    actual_direction.set_offsets([[robot.x, robot.y]])

    actual_direction.set_UVC(np.cos(robot.orientation), np.sin(robot.orientation))
    
    
    return (actual_position, estimated_position, estimated_landmarks,
            actual_path, estimated_path, actual_direction)

# 애니메이션 생성 및 저장
anim = animation.FuncAnimation(fig, animate, frames=50,
                             interval=550, init_func=init, blit=True)

# 애니메이션 저장 시도


try:

    from matplotlib.animation import PillowWriter

    writer = PillowWriter(fps=20)

    anim.save('slam.gif', writer=writer)


except Exception as e:

    print(f"Error saving animation: {e}")

    plt.show()  # 저장 실패 시 화면에 표시