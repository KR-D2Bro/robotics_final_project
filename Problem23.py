import random


import numpy as np
from math import *
from copy import deepcopy
import matplotlib.pyplot as plt

WORLD_SIZE_X = 25   #세로
WORLD_SIZE_Y = 15   #가로


#맵 생성. 0은 빈칸, 1은 벽
#                0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
MAP = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   #0
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   #1
                 [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],   #2
                 [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],   #3
                 [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],   #4
                 [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1],   #5
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1],   #6
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],   #7
                 [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],   #8
                 [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   #9
                 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   #10
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],   #11
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0],   #12
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   #13
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])   #14


heuristic_p1 = np.array([[14, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13],   #0
                        [14, 13, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],   #1
                        [14, 13, 12, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11],   #2
                        [14, 13, 12, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],   #3
                        [14, 13, 12, 11, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10],   #4
                        [14, 13, 12, 11, 10, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 10],   #5
                        [14, 13, 12, 11, 10, 9, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 9, 10],   #6
                        [14, 13, 12, 11, 10, 9, 8, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 8, 9, 10],   #7
                        [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 7, 8, 9, 10],   #8
                        [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6, 7, 8, 9, 10],   #9
                        [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10],   #10
                        [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 2, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10],   #11
                        [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],   #12
                        [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],   #13
                        [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])  #14


landmarks = [[3.5, 5.5],[5.5, 16.5],[10.5, 5.5],[12.5, 15.5]]
P1 = [13,14]




DISTANCE_NOISE_STD = 0.1
SENSE_NOISE_STD = 0.4
STEERING_NOISE_STD = 0.1


# ------------------------------------------------

#

# this is the Robot class

#


class robot:

  # Initialize the position and orientation of the robot

  def __init__(self, length = 1):

    self.x = random.uniform(0,WORLD_SIZE_X)

    self.y = random.uniform(0, WORLD_SIZE_Y)

    self.orientation = 0.0

    self.length = length

    self.steering_noise = 0.0

    self.distance_noise = 0.0

    self.steering_drift = 0.0

    self.sense_noise   = 0.0;

  def set(self, new_x, new_y, new_orientation=2 * np.pi): #위치 제한

    self.x = float(new_x)

    self.y = float(new_y)

    self.orientation = float(new_orientation)

  def set_noise(self, steering_noise, distance_noise, sense_noise):

    self.steering_noise = steering_noise

    self.distance_noise = distance_noise

    self.sense_noise = sense_noise


  def sense(self): #랜드마크와 거리측

    Z = []

    for i in range(len(landmarks)):

      dist = sqrt((self.x - landmarks[i][1]) ** 2 + (self.y - landmarks[i][0]) ** 2)

      if dist > 5:
        dist = 5

      dist += random.gauss(0.0, self.sense_noise)

      Z.append(dist)

    return Z
  
  def move(self, steering, isReal = True , distance=1, tolerance=0.001, max_steering_angle=np.pi/4): 

    """

    steering = front wheel steering angle, limited by max_steering_angle

    distance = total distance driven, most be non-negative

    """
    i=1

    while True:# apply noise
      steering2 = steering + random.gauss(0.0, self.steering_noise) # steering = 알파 (바이시클 모델)

      #핸들의 허용 각도가 넘어가면 최대 각도로 설정
      if steering2 >  max_steering_angle:

        steering2 = max_steering_angle

      if steering2 < -max_steering_angle:

        steering2 = -max_steering_angle

      distance2 = random.gauss(distance, self.distance_noise) # distance = d (바이시클 모델)

      # Execute motion

      turn = np.tan(steering2) * distance2 / self.length # turn = 베타(바이시클모델)

      if abs(turn) < tolerance:  # 움직임이 미세할 경우 바이시클 모델이 아닌 점이 움직이는 걸로 가정하고 계산

        # approximate by straight line motion

        x = self.x + distance2 * np.cos(self.orientation)

        y = self.y + distance2 * np.sin(self.orientation)

        orientation = (self.orientation + turn) % (2.0 * np.pi)

      else:    # 움직임이 적당히 클 경우
        # approximate bicycle model for motion
        radius = distance2 / turn # = 회전 반경

        # 회전 중심 위치
        cx = self.x - (np.sin(self.orientation) * radius)

        cy = self.y + (np.cos(self.orientation) * radius)

        # 다음 상태의 위치 + 자세
        orientation = (self.orientation + turn) % (2.0 * np.pi) # = 세타 + 베타

        x = cx + (np.sin(orientation) * radius)

        y = cy - (np.cos(orientation) * radius)
      
      # 장애물 체크
      #장애물이 없다면 통과, 있다면 orientation을 조정 후 다른 path진행
      if not (isReal and (x<0 or x>= WORLD_SIZE_X or y<0 or y>=WORLD_SIZE_Y or MAP[int(y)][int(x)] == 1) ):
        break
      self.orientation -= np.pi * i / 50
      i+=1

    res = robot()

    res.set(x, y, orientation)

    res.set_noise(self.steering_noise, self.distance_noise, self.sense_noise)

    return res

  def p_move(self, steering, isReal = True , distance=1, tolerance=0.001, max_steering_angle=np.pi/4): 

    """

    steering = front wheel steering angle, limited by max_steering_angle

    distance = total distance driven, most be non-negative

    """

    i=1

    while True:# apply noise
      steering2 = steering + random.gauss(0.0, self.steering_noise) # steering = 알파 (바이시클 모델)

      #핸들의 허용 각도가 넘어가면 최대 각도로 설정
      if steering2 >  max_steering_angle:

        steering2 = max_steering_angle

      if steering2 < -max_steering_angle:

        steering2 = -max_steering_angle

      distance2 = random.gauss(distance, self.distance_noise) # distance = d (바이시클 모델)

      # Execute motion

      turn = np.tan(steering2) * distance2 / self.length # turn = 베타(바이시클모델)

      if abs(turn) < tolerance:  # 움직임이 미세할 경우 바이시클 모델이 아닌 점이 움직이는 걸로 가정하고 계산

        # approximate by straight line motion

        x = self.x + distance2 * np.cos(self.orientation)

        y = self.y + distance2 * np.sin(self.orientation)

        orientation = (self.orientation + turn) % (2.0 * np.pi)

      else:    # 움직임이 적당히 클 경우
        # approximate bicycle model for motion
        radius = distance2 / turn # = 회전 반경

        # 회전 중심 위치
        cx = self.x - (np.sin(self.orientation) * radius)

        cy = self.y + (np.cos(self.orientation) * radius)

        # 다음 상태의 위치 + 자세
        orientation = (self.orientation + turn) % (2.0 * np.pi) # = 세타 + 베타

        x = cx + (np.sin(orientation) * radius)

        y = cy - (np.cos(orientation) * radius)
      
      # 장애물 체크
      #장애물이 없다면 통과, 있다면 orientation을 조정 후 다른 path진행
      if not (isReal and (x<0 or x>= WORLD_SIZE_X or y<0 or y>=WORLD_SIZE_Y) ):
        break
      self.orientation -= np.pi * i / 50
      i+=1

    res = robot()

    res.set(x, y, orientation)

    res.set_noise(self.steering_noise, self.distance_noise, self.sense_noise)

    return res

  def Gaussian(self, mu, sigma, x):

    return exp(-((mu-x)**2)/(sigma**2)/2)/sqrt(2.0*pi*(sigma**2))  #로봇의 측정 거리와 샘플의 측정거리를 비교 + 차이가 클수록 가우시안의 외각으로 빠짐-> 값이 작아짐.

  def measure_prob(self, measure, sigma = 2.0):

    prob =1.0

    for i in range(len(landmarks)):

      dist = sqrt((self.x-landmarks[i][1])**2 + (self.y-landmarks[i][0])**2)

      if dist > 5:
        dist = 5

        
      dist += random.gauss(0.0, self.sense_noise)

      prob *= self.Gaussian(dist, sigma, measure[i] )

    return prob

def eval(r,p):

  sum = 0.0

  for i in range(len(p)):

    dx = (p[i].x - r.x )

    dy = (p[i].y - r.y )

    err = sqrt(dx * dx + dy * dy)

    sum += err

  return sum/float(len(p))

def plot(myRobot = None, p = None, t= None, title = None, plan_path = None, footprint=None):
    plt.figure()

    # MAP의 1에 해당하는 부분을 검은색 네모 박스로 추가
    plt.scatter([], [], marker='s', color='blue', alpha= 0.3, s=100, label='Block')
    plt.scatter([], [], marker='s', color='black', alpha= 0.5, s=100, label='Parking space')
    for i in range(WORLD_SIZE_Y):
        for j in range(WORLD_SIZE_X):
            if MAP[i, j] == 1:  # 벽인 경우
              plt.gca().add_patch(plt.Rectangle((j, i), 1, 1, color='blue', alpha = 0.3))  # 사각형 추가
            
    #주차
    plt.gca().add_patch(plt.Rectangle((P1[1], P1[0]), 1, 1, color='black', alpha = 0.5))  # 사각형 추가
    # 랜드마크 표시
    plt.scatter([], [], marker='s', color='blue', s=100, label='Landmarks')
    for l in landmarks:
      plt.gca().add_patch(plt.Rectangle((l[1] - 0.5, l[0] - 0.5), 1, 1, color='blue', alpha=0.9))  # 1x1 크기의 사각형 추가

    # 파티클 표시
    if p is not None:
      #전체 샘플들의 위치 표시
      plt.scatter([particle.x for particle in p], [particle.y for particle in p], color='green', alpha=0.3, label='Particles')
      #샘플들의 평균 추정 위치 표시
      estimated_position = [0.0, 0.0]
      for e in p:
        estimated_position[0] += e.x
        estimated_position[1] += e.y
      estimated_position[0] /= N
      estimated_position[1] /= N
      plt.scatter(estimated_position[0], estimated_position[1], color='orange', alpha=1.0, s=100, label='Estimated position')
      
          # 로봇의 실제 위치 표시
    if myRobot is not None:
      plt.scatter(myRobot.x, myRobot.y, marker='o', color='blue', s=100, label='True Robot Position')
      
    # 경로 표시
    if plan_path is not None:
      plt.plot(plan_path[:,1],plan_path[:,0],'r.-', label='Path')
    
    if footprint is not None:
      x_coords, y_coords = zip(*footprint)
      plt.plot(x_coords,y_coords,'b.-', label='Footprint')

    plt.xlim(0, WORLD_SIZE_X)  # x축 범위 설정
    plt.ylim(WORLD_SIZE_Y, 0)  # y축 범위 설정 (위에서 아래로 증가)
    plt.xlabel('X')
    plt.ylabel('Y')
    if title != None and t != None:
      plt.title(title + str(t))
    elif title != None:
      plt.title(title)
    plt.legend()
    plt.show()


#문제 2 시작-------------------------------------------------------------------------------------
def particle(myRobot, p, steering = random.uniform(-1 * np.pi /4, np.pi/ 4), distance=1, N=1000, plan_path = None, count = None, sigma = 0.7, footprint = None):

  myRobot = myRobot.move(steering, distance=distance) #실제 로봇 move

  # for e in p:
  #   e.set(e.x,e.y,random.uniform(0, 2*pi))

  p = [e.move(steering, distance = distance) for e in p] #샘플 로봇 move

  Z = myRobot.sense()

  if plan_path is not None and count is not None:
    plot(myRobot = myRobot, p= p, plan_path= spath, title= "Moving", t = count, footprint=footprint)

  else:
    plot(myRobot = myRobot, p = p, t = t, title = "step ")

    print('Localization error: ', eval(myRobot, p))

  w = [e.measure_prob(Z, sigma) for e in p]

  # Resampling

  p3 = []  # 새로운 파티클 집합을 저장할 리스트 초기화

  index = int(random.random() * N)  # 초기 인덱스를 랜덤하게 선택

  beta = 0.0  # 베타 변수 초기화

  mw = max(w)  # 현재 파티클의 가중치 중 최대값을 mw에 저장

  for i in range(N):  # N개의 파티클에 대해 반복
    beta += random.random() * 2.0 * mw  # 베타 값을 업데이트 (0에서 2*mw 사이의 랜덤 값 추가)

    while beta > w[index]:  # 베타 값이 현재 파티클의 가중치보다 크면
        beta -= w[index]  # 베타에서 현재 파티클의 가중치를 빼고
        index = (index + 1) % N  # 인덱스를 다음 파티클로 이동 (순환)

    p3.append(p[index])  # 선택된 파티클을 새로운 리스트에 추가

  return myRobot, p3 # 새로운 파티클 집합으로 업데이트


myRobot = robot()

#위치가 1.5, 1.5이라고 가정
myRobot.set(1.5,1.5)

myRobot.set_noise(STEERING_NOISE_STD, DISTANCE_NOISE_STD, SENSE_NOISE_STD)

N = 1000

T = 20

# 1000 Partilces are initialized

p = []

for i in range(N):

  r = robot()

  r.set(r.x,r.y,random.uniform(0, 2*pi))

  r.set_noise(STEERING_NOISE_STD,DISTANCE_NOISE_STD,SENSE_NOISE_STD)

  p.append(r)

#무작위 움직임에 대한 파티클 필터 실행 => localization
t=0
while eval(myRobot,p) >= 1.0:
   myRobot, p = particle(myRobot, p, N= N)
   t+=1
  
print('Localization error: ', eval(myRobot, p))
#문제 2 끝 --------------------------------------------------------------------------------------

#문제 3 시작-------------------------------------------------------------------------------------
estimated_position = [0.0, 0.0]

for e in p:
   estimated_position[0] += e.x
   estimated_position[1] += e.y

estimated_position[0] /= N
estimated_position[1] /= N

init = [int(estimated_position[1]),int(estimated_position[0])]

print("estimated position : (%f, %f)"% (estimated_position[0], estimated_position[1] ))
print("real position : (%f, %f)"% (myRobot.x, myRobot.y))

goal = P1 # (13,14)

delta = [[-1, 0 ], # go up

         [ 0, -1], # go left

         [ 1, 0 ], # go down

         [ 0, 1 ]] # go right

delta_name = ['^', '<', 'v', '>']

cost = 1

plan_path = np.empty((0, 2), dtype=float)  # 초기화: 빈 2차원 배열

def search():

    closed = [[0 for row in range(len(MAP[0]))] for col in range(len(MAP))]   # 확장 여부 확인

    closed[init[0]][init[1]] = 1

    #print(closed);

    action = [[-1 for row in range(len(MAP[0]))] for col in range(len(MAP))]  # action index 표시  (delta)


    x = init[0]

    y = init[1]

    g = 0

    # A* 알고리즘에서는 g + h값을 최소로 하는 노드 탐색

    f = g + heuristic_p1[x, y]  # f = g + h

    m = -1

    # g 값을 사용해야하므로 open 배열에 g값과 h값을 둘 다 저장하는 형식 사용
    open = [[ f, g, x, y, m]]

    found = False  # flag that is set when search is complet

    resign = False # flag set if we can't find expand

    count = 1

    while not found and not resign:

        if len(open) == 0:

            resign = True

        else:

            open.sort()

            open.reverse()

            next = open.pop()

            x = next[2]

            y = next[3]

            g = next[1]

            m = next[4]

            #print("node to expand")

            #print(next)

            action[x][y] = m

            if x == goal[0] and y == goal[1]:

                found = True

            else:

                for i in range(len(delta)):

                    x2 = x + delta[i][0]

                    y2 = y + delta[i][1]

                    if x2 >= 0 and x2 < len(MAP) and y2 >=0 and y2 < len(MAP[0]):

                        if closed[x2][y2] == 0 and MAP[x2, y2] == 0:

                            g2 = g + cost

                            # 해당 노드의 g값과 h값을 더한 f값 저장

                            f2 = g2 + heuristic_p1[x2, y2]

                            m2 = i

                            open.append([ f2, g2, x2, y2, m2])

                            closed[x2][y2] = 1

                            count += 1

    #plan_path : action으로 얻은 움직임으로부터 경로 생성
    if found :
        global plan_path
        x = goal[0]
        y = goal[1]
        while x!=init[0] or y != init[1]:
            delta_x = delta[action[x][y]][0]
            delta_y = delta[action[x][y]][1]

            plan_path = np.insert(plan_path, 0,[[x+0.5,y+0.5]], axis=0)

            x = x - delta_x;
            y = y - delta_y;
        plan_path = np.insert(plan_path, 0,[[x+0.5,y+0.5]], axis=0)

def smoothing(plan_path, weight_data = 0.5, weight_smooth = 0.7, tolerance = 0.0001):
  # Make a deep copy of path into newpath

  newpath = deepcopy(plan_path)

  error1=10000

  error2=0

  while (abs(error1-error2) >= tolerance):

    error1=error2

    error2=0

    for i in range(1,len(plan_path)-1):

      d0 = [0.0, 0.0]

      for k in range(2):

        d1=weight_data*(plan_path[i,k]-newpath[i,k])

        d2=weight_smooth*(newpath[i+1,k]+newpath[i-1,k]-2*newpath[i,k])

        d0[k] = d1 + d2

        newpath[i,k]=newpath[i,k]+d1+d2

        error2 += d1**2+d2**2
      
      while MAP[int(newpath[i, 0]), int(newpath[i,1])] == 1:
         newpath[i,0] -= d0[0]/10
         newpath[i,1] -= d0[1]/10

  return newpath

search()

plot(myRobot= myRobot, p = p, plan_path= plan_path, title= "Path with A*")

spath=smoothing(plan_path)

plot(myRobot= myRobot, p= p, plan_path = spath, title = "Path after smoothing")

#문제 3 smoothing 끝-------------------------------------------------------------------------------------

#문제 3 PID 시작-----------------------------------------------------------------------------------


def run_PID_follow_path(real_robot, p, spath, tau_p, tau_d, tau_i, n=100, tolerance=0.2,footprint = None):
    cte_sum = 0  # 적분 항
    cte_prev = 0  # 이전 CTE
    cte = 0

    real_robot.set(real_robot.x, real_robot.y, np.arctan2(spath[1][0]-spath[0][0], spath[1][1]-spath[0][1]))

    estimated_position = [0.0, 0.0]

    for e in p:
      estimated_position[0] += e.x
      estimated_position[1] += e.y
      e.orientation = real_robot.orientation

    estimated_position[0] /= N
    estimated_position[1] /= N

    estimated_robot = robot()
    estimated_robot.set(estimated_position[0],estimated_position[1], real_robot.orientation)

    footprint.append([real_robot.x, real_robot.y])

    for i in range(n):
        # 추정 위치가 P1에 도착하면 중단
        if int(estimated_robot.x) == P1[1] and int(estimated_robot.y) == P1[0]:
            break
        # 현재 로봇 위치 가져오기
        current_position = np.array([estimated_robot.y, estimated_robot.x])  # [y, x] 형식

        # spath에서 가장 가까운 점 찾기
        closest_point = min(spath, key=lambda point: np.linalg.norm(current_position - np.array(point)))
        closest_point = np.array(closest_point)

        # closest_point의 인덱스를 찾기 위해 np.where 사용
        closest_index = np.where((spath == closest_point).all(axis=1))[0][0]  # closest_index 정의

        if closest_index < len(spath) - 2:
            next_point = np.array(spath[closest_index + 2])

            line_vector = next_point - closest_point
            # CTE 계산
            direction_vector = closest_point - current_position
            cte = np.cross(line_vector, direction_vector) / np.linalg.norm(line_vector)  # CTE 계산 개선


            # PID 제어 계산
            cte_sum += cte  # 적분 값 업데이트
            cte_diff = cte - cte_prev  # 미분 값
            steering = -tau_p * cte - tau_d * cte_diff - tau_i * cte_sum  # PID 제어
            cte_prev = cte  # 이전 CTE 갱신

            # 조향각 에러 추가
            yaw_error = normalize_angle(np.arctan2(line_vector[0], line_vector[1]) - normalize_angle(estimated_robot.orientation))
            steering = steering * 0.9 + yaw_error * 1.3  # 조향각 에러에 가중치 추가 (가중치 조정)

            # 로봇 이동
            # robot , p = particle(robot, p, steering = steering, distance=0.5)
            real_robot, p = particle(real_robot, p, steering = steering, distance=0.5, N= N, plan_path = spath, count = i, sigma = 0.3,footprint = footprint)

        else:
            prev_point = np.array(spath[closest_index - 1])

            line_vector = closest_point - prev_point
            direction_vector = closest_point - current_position
            cte = np.cross(line_vector, direction_vector) / np.linalg.norm(line_vector)  # CTE 계산 개선
            # 경로의 왼쪽 또는 오른쪽 판단

            # PID 제어 계산
            cte_sum += cte  # 적분 값 업데이트
            cte_diff = cte - cte_prev  # 미분 값
            steering = -tau_p * cte - tau_d * cte_diff - tau_i * cte_sum  # PID 제어
            cte_prev = cte  # 이전 CTE 갱신

            # 조향각 에러 추가
            yaw_error = normalize_angle(np.arctan2(line_vector[0], line_vector[1]) - normalize_angle(estimated_robot.orientation))
            steering = steering * 1.3 + yaw_error * 0.5  # 조향각 에러에 가중치 추가 (가중치 조정)

            real_robot, p = particle(real_robot, p, steering = steering, distance=0.5, N= N, plan_path = spath, count = i, sigma = 0.3,footprint = footprint)

            # 경로 시각화 (선택적으로 주기 조정 가능)
            # plot(myRobot= robot, p = p, plan_path= spath, title = "Moving Step", t=i)

            # print('Localization error: ', eval(robot, p))

            # p = resampling(robot,p);
        estimated_position = [0.0, 0.0]

        for e in p:
          estimated_position[0] += e.x
          estimated_position[1] += e.y

        estimated_position[0] /= N
        estimated_position[1] /= N

        estimated_robot.set(estimated_position[0],estimated_position[1], estimate_orientation(p))

        footprint.append([real_robot.x, real_robot.y])


        

def normalize_angle(angle):
    """Normalize the angle to be within the range [-pi, pi]."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

def estimate_orientation(particles):
    sin_sum = 0.0
    cos_sum = 0.0 
    N = len(particles) # 파티클의 개수 
    for particle in particles: 
      sin_sum += np.sin(particle.orientation) 
      cos_sum += np.cos(particle.orientation) 
      mean_orientation = np.arctan2(sin_sum / N, cos_sum / N) 
    return mean_orientation

Nt=300

footprint =[]

run_PID_follow_path(myRobot, p, spath, 0.3, 4, 0.01, Nt, footprint= footprint)
