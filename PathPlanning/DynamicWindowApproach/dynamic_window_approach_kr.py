"""
Mobile robot motion planning sample with Dynamic Window Approach

author: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı

Dynamic Window Approach (DWA)는 로봇의 이동 경로를 실시간으로 계획하는 알고리즘입니다.
이 코드는 로봇의 현재 상태, 목표 위치, 장애물 위치를 고려하여 최적의 속도와 방향을 계산합니다.
"""

import math
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np

show_animation = True  # 시뮬레이션 실행 시 애니메이션을 표시할지 여부

# ------------------------ Core Functions ------------------------

def dwa_control(x, config, goal, ob):
    """
    Dynamic Window Approach control
    현재 로봇 상태와 설정, 목표 위치, 장애물 정보를 기반으로 최적의 제어 입력을 계산합니다.

    Parameters:
        x (array): 현재 상태 [x 위치, y 위치, 방향 각도, 선속도, 각속도]
        config (Config): 로봇 및 환경 설정 정보
        goal (array): 목표 위치 [x, y]
        ob (array): 장애물 위치 리스트

    Returns:
        u (array): 최적의 제어 입력 [선속도, 각속도]
        trajectory (array): 예측된 최적 경로
    """
    dw = calc_dynamic_window(x, config)  # Dynamic Window 계산
    u, trajectory = calc_control_and_trajectory(x, dw, config, goal, ob)  # 최적 제어 입력 계산
    return u, trajectory


# ------------------------ Classes ------------------------

class RobotType(Enum):
    """
    로봇 형태를 정의하는 Enum 클래스
    """
    circle = 0  # 원형 로봇
    rectangle = 1  # 직사각형 로봇


class Config:
    """
    시뮬레이션 및 로봇 설정 정보를 담는 클래스
    """
    def __init__(self):
        # 로봇 제약 조건
        self.max_speed = 1.0  # 최대 선속도 [m/s]
        self.min_speed = -0.5  # 최소 선속도 [m/s]
        self.max_yaw_rate = 40.0 * math.pi / 180.0  # 최대 각속도 [rad/s]
        self.max_accel = 0.2  # 최대 가속도 [m/ss]
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # 최대 각속도 변화 [rad/ss]
        self.v_resolution = 0.01  # 속도 샘플링 간격 [m/s]
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0  # 각속도 샘플링 간격 [rad/s]
        self.dt = 0.1  # 시간 간격 [s]
        self.predict_time = 3.0  # 경로 예측 시간 [s]
        self.to_goal_cost_gain = 0.15  # 목표 비용 가중치
        self.speed_cost_gain = 1.0  # 속도 비용 가중치
        self.obstacle_cost_gain = 1.0  # 장애물 비용 가중치
        self.robot_stuck_flag_cons = 0.001  # 로봇 정체 방지 상수
        self.robot_type = RobotType.circle  # 기본 로봇 형태: 원형

        # 로봇 충돌 체크 및 크기
        self.robot_radius = 1.0  # 원형 로봇의 반경 [m]
        self.robot_width = 0.5  # 직사각형 로봇의 너비 [m]
        self.robot_length = 1.2  # 직사각형 로봇의 길이 [m]

        # 장애물 정보 (고정)
        self.ob = np.array([[-1, -1],
                            [0, 2],
                            [4.0, 2.0],
                            [5.0, 4.0],
                            [5.0, 5.0],
                            [5.0, 6.0],
                            [5.0, 9.0],
                            [8.0, 9.0],
                            [7.0, 9.0],
                            [8.0, 10.0],
                            [9.0, 11.0],
                            [12.0, 13.0],
                            [12.0, 12.0],
                            [15.0, 15.0],
                            [13.0, 13.0]])

# ------------------------ Utility Functions ------------------------

def motion(x, u, dt):
    """
    로봇 운동 모델
    입력 속도와 시간 간격에 따라 새로운 상태를 계산합니다.

    Parameters:
        x (array): 현재 상태 [x 위치, y 위치, 방향 각도, 선속도, 각속도]
        u (array): 제어 입력 [선속도, 각속도]
        dt (float): 시간 간격 [s]

    Returns:
        array: 업데이트된 상태
    """
    x[2] += u[1] * dt  # 방향 각도 업데이트
    x[0] += u[0] * math.cos(x[2]) * dt  # x 위치 업데이트
    x[1] += u[0] * math.sin(x[2]) * dt  # y 위치 업데이트
    x[3] = u[0]  # 선속도 업데이트
    x[4] = u[1]  # 각속도 업데이트
    return x


def calc_dynamic_window(x, config):
    """
    동적 윈도우 계산
    로봇의 속도 및 가속도 제약 조건과 현재 상태를 기반으로 가능한 속도/각속도 범위를 계산합니다.

    Parameters:
        x (array): 현재 상태 [x 위치, y 위치, 방향 각도, 선속도, 각속도]
        config (Config): 로봇 설정 정보

    Returns:
        array: 동적 윈도우 [v_min, v_max, yaw_rate_min, yaw_rate_max]
    """
    # 로봇 사양에 따른 동적 윈도우
    Vs = [config.min_speed, config.max_speed, -config.max_yaw_rate, config.max_yaw_rate]

    # 운동 모델에 따른 동적 윈도우
    Vd = [x[3] - config.max_accel * config.dt, x[3] + config.max_accel * config.dt,
          x[4] - config.max_delta_yaw_rate * config.dt, x[4] + config.max_delta_yaw_rate * config.dt]

    # 최종 동적 윈도우
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw

def predict_trajectory(x_init, v, y, config):
    """
    주어진 속도와 각속도를 기반으로 로봇의 경로를 예측합니다.

    Parameters:
        x_init (array): 초기 상태 [x 위치, y 위치, 방향 각도, 선속도, 각속도]
        v (float): 선속도 [m/s]
        y (float): 각속도 [rad/s]
        config (Config): 로봇 설정 정보

    Returns:
        trajectory (array): 예측된 경로
    """
    x = np.array(x_init)  # 초기 상태 복사
    trajectory = np.array(x)  # 경로 저장
    time = 0  # 시뮬레이션 시간 초기화

    # 주어진 예측 시간 동안 경로 계산
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)  # 시간 단위로 로봇 상태 업데이트
        trajectory = np.vstack((trajectory, x))  # 새로운 상태를 경로에 추가
        time += config.dt  # 시간 증가

    return trajectory


def calc_control_and_trajectory(x, dw, config, goal, ob):
    """
    동적 윈도우 안에서 모든 가능한 제어 입력을 평가해 최적의 입력과 경로를 찾습니다.

    Parameters:
        x (array): 현재 상태 [x 위치, y 위치, 방향 각도, 선속도, 각속도]
        dw (array): 동적 윈도우 [v_min, v_max, yaw_rate_min, yaw_rate_max]
        config (Config): 로봇 설정 정보
        goal (array): 목표 위치 [x, y]
        ob (array): 장애물 리스트

    Returns:
        best_u (array): 최적의 제어 입력 [선속도, 각속도]
        best_trajectory (array): 최적 경로
    """
    x_init = x[:]  # 초기 상태 저장
    min_cost = float("inf")  # 최소 비용 초기화
    best_u = [0.0, 0.0]  # 최적 제어 입력 초기화
    best_trajectory = np.array([x])  # 최적 경로 초기화

    # 동적 윈도우 내 속도와 각속도 샘플링
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):
            trajectory = predict_trajectory(x_init, v, y, config)  # 경로 예측

            # 비용 계산
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
            speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
            ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, ob, config)
            final_cost = to_goal_cost + speed_cost + ob_cost

            # 최소 비용 경로 및 제어 입력 업데이트
            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v, y]
                best_trajectory = trajectory

                # 로봇이 멈추지 않도록 설정
                if abs(best_u[0]) < config.robot_stuck_flag_cons \
                        and abs(x[3]) < config.robot_stuck_flag_cons:
                    best_u[1] = -config.max_delta_yaw_rate  # 각속도 변경

    return best_u, best_trajectory


def calc_obstacle_cost(trajectory, ob, config):
    """
    장애물 비용을 계산합니다. 충돌 시 무한대 비용 반환.

    Parameters:
        trajectory (array): 로봇의 예측 경로
        ob (array): 장애물 위치 리스트
        config (Config): 로봇 설정 정보

    Returns:
        float: 장애물 비용
    """
    ox = ob[:, 0]  # 장애물 x 좌표
    oy = ob[:, 1]  # 장애물 y 좌표
    dx = trajectory[:, 0] - ox[:, None]  # 경로 x와 장애물 간 거리
    dy = trajectory[:, 1] - oy[:, None]  # 경로 y와 장애물 간 거리
    r = np.hypot(dx, dy)  # 거리 계산

    if config.robot_type == RobotType.rectangle:  # 직사각형 로봇일 경우
        yaw = trajectory[:, 2]
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        rot = np.transpose(rot, [2, 0, 1])  # 회전 행렬 생성
        local_ob = ob[:, None] - trajectory[:, 0:2]
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        local_ob = np.array([local_ob @ x for x in rot])
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        upper_check = local_ob[:, 0] <= config.robot_length / 2
        right_check = local_ob[:, 1] <= config.robot_width / 2
        bottom_check = local_ob[:, 0] >= -config.robot_length / 2
        left_check = local_ob[:, 1] >= -config.robot_width / 2
        if (np.logical_and(np.logical_and(upper_check, right_check),
                           np.logical_and(bottom_check, left_check))).any():
            return float("Inf")  # 충돌 발생 시 무한대 반환
    elif config.robot_type == RobotType.circle:  # 원형 로봇일 경우
        if np.array(r <= config.robot_radius).any():
            return float("Inf")  # 충돌 발생 시 무한대 반환

    min_r = np.min(r)  # 가장 가까운 장애물까지의 거리
    return 1.0 / min_r  # 장애물 비용 반환


def calc_to_goal_cost(trajectory, goal):
    """
    목표까지의 각도 차이를 기반으로 목표 비용 계산.

    Parameters:
        trajectory (array): 로봇의 예측 경로
        goal (array): 목표 위치 [x, y]

    Returns:
        float: 목표 비용
    """
    dx = goal[0] - trajectory[-1, 0]  # 목표까지의 x 거리
    dy = goal[1] - trajectory[-1, 1]  # 목표까지의 y 거리
    error_angle = math.atan2(dy, dx)  # 목표 방향 각도
    cost_angle = error_angle - trajectory[-1, 2]  # 로봇 방향과 목표 방향의 차이
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))  # 각도 차이 계산
    return cost


def plot_arrow(x, y, yaw, length=0.5, width=0.1):
    """
    로봇 방향을 화살표로 시각화합니다.

    Parameters:
        x (float): x 위치
        y (float): y 위치
        yaw (float): 방향 각도 [rad]
        length (float): 화살표 길이
        width (float): 화살표 너비
    """
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)


def plot_robot(x, y, yaw, config):
    """
    로봇을 시각화합니다.

    Parameters:
        x (float): x 위치
        y (float): y 위치
        yaw (float): 방향 각도 [rad]
        config (Config): 로봇 설정 정보
    """
    if config.robot_type == RobotType.rectangle:  # 직사각형 로봇
        outline = np.array([[-config.robot_length / 2, config.robot_length / 2,
                             config.robot_length / 2, -config.robot_length / 2,
                             -config.robot_length / 2],
                            [config.robot_width / 2, config.robot_width / 2,
                             -config.robot_width / 2, -config.robot_width / 2,
                             config.robot_width / 2]])
        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                         [-math.sin(yaw), math.cos(yaw)]])
        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y
        plt.plot(outline[0, :], outline[1, :], "-k")
    elif config.robot_type == RobotType.circle:  # 원형 로봇
        circle = plt.Circle((x, y), config.robot_radius, color="b")
        plt.gcf().gca().add_artist(circle)
        out_x, out_y = (np.array([x, y]) +
                        np.array([np.cos(yaw), np.sin(yaw)]) * config.robot_radius)
        plt.plot([x, out_x], [y, out_y], "-k")

def main(gx=10.0, gy=10.0, robot_type=RobotType.circle):
    """
    시뮬레이션을 실행하고 결과를 시각화합니다.

    Parameters:
        gx (float): 목표 x 좌표
        gy (float): 목표 y 좌표
        robot_type (RobotType): 로봇의 형태 (circle 또는 rectangle)
    """
    print("Simulation started!")  # 시작 메시지 출력

    # Config 객체 생성
    config = Config()  # Config 객체 초기화
    config.robot_type = robot_type  # 로봇 형태 설정

    # 초기 상태 정의: [x 위치, y 위치, 방향 각도, 선속도, 각속도]
    x = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])
    # 목표 위치 정의: [x, y]
    goal = np.array([gx, gy])

    trajectory = np.array(x)  # 로봇 경로 기록 초기화
    ob = config.ob  # 장애물 리스트 가져오기

    while True:
        # DWA 제어를 통해 최적의 입력과 경로 계산
        u, predicted_trajectory = dwa_control(x, config, goal, ob)
        x = motion(x, u, config.dt)  # 로봇의 새로운 상태 계산
        trajectory = np.vstack((trajectory, x))  # 현재 상태를 경로에 추가

        if show_animation:
            # 애니메이션 업데이트
            plt.cla()  # 현재 그림 지우기
            # ESC 키로 시뮬레이션 종료
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")  # 예측 경로
            plt.plot(x[0], x[1], "xr")  # 현재 로봇 위치
            plt.plot(goal[0], goal[1], "xb")  # 목표 위치
            plt.plot(ob[:, 0], ob[:, 1], "ok")  # 장애물
            plot_robot(x[0], x[1], x[2], config)  # 로봇의 현재 상태 시각화
            plot_arrow(x[0], x[1], x[2])  # 로봇 방향 표시
            plt.axis("equal")  # 축 비율 고정
            plt.grid(True)  # 그리드 활성화
            plt.pause(0.0001)  # 짧은 시간 대기

        # 목표 도달 여부 확인
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])  # 목표와의 거리
        if dist_to_goal <= config.robot_radius:
            print("Goal!!")  # 목표 도달 메시지 출력
            break

    print("Done")  # 시뮬레이션 완료 메시지
    if show_animation:
        # 전체 경로 시각화
        plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")  # 로봇 이동 경로
        plt.pause(0.0001)
        plt.show()

if __name__ == '__main__':
    # 시뮬레이션 실행
    # 로봇 형태를 바꾸어 실행할 수 있음 (rectangle 또는 circle)
    main(robot_type=RobotType.rectangle)
    # main(robot_type=RobotType.circle)