import sys
import time

import rclpy  # ROS2 Python客户端库
from rclpy.node import Node  # ROS2节点基类
from rclpy.signals import SignalHandlerOptions  # 信号处理选项
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor  # 执行器和外部关闭异常
from rclpy.qos import QoSPresetProfiles  # QoS预设配置
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup  # 回调组，用于控制并发执行

from std_msgs.msg import Float32  # 标准浮点消息
from geometry_msgs.msg import Twist, Pose  # 几何消息，用于速度和位姿
from nav_msgs.msg import Odometry  # 里程计消息
from sensor_msgs.msg import LaserScan  # 激光扫描数据消息

from assessment_interfaces.msg import Item, ItemList  # 自定义消息类型
from auro_interfaces.msg import StringWithPose
from auro_interfaces.srv import ItemRequest  # 自定义服务类型

from tf_transformations import euler_from_quaternion  # 四元数到欧拉角转换
import angles  # 角度计算工具
from enum import Enum  # 枚举类型，用于定义状态
import random  # 随机数生成器
import math  # 数学函数库

# 定义机器人运动参数常量
LINEAR_VELOCITY = 0.5  # 机器人前进速度（米/秒）
ANGULAR_VELOCITY = 0.5  # 机器人转向速度（弧度/秒）

# 定义转向方向常量
TURN_LEFT = 1  # 向左转的角速度方向
TURN_RIGHT = -1  # 向右转的角速度方向

# 定义激光雷达参数
SCAN_THRESHOLD = 0.2  # 激光雷达检测距离阈值（米）
SCAN_FRONT = 0  # 前方区域索引
SCAN_LEFT = 1  # 左侧区域索引
SCAN_BACK = 2  # 后方区域索引
SCAN_RIGHT = 3  # 右侧区域索引

# 定义机器人状态
class State(Enum):
    FORWARD = 0  # 向前行驶状态
    TURNING = 1  # 转向状态
    COLLECTING = 2  # 收集物品状态
    MOVING_TO_DROP_POINT = 3  # 前往放置物品的固定位置状态
    DROPPING = 4  # 放置物品状态

class RobotController(Node):
    def __init__(self):
        # 初始化节点
        super().__init__('robot_controller')

        # 初始化执行器引用
        self.executor = None

        # 初始化机器人状态变量
        self.state = State.FORWARD  # 初始状态为向前行驶
        self.pose = Pose()  # 当前位姿
        self.previous_pose = Pose()  # 上一次位姿
        self.yaw = 0.0  # 当前朝向角度
        self.previous_yaw = 0.0  # 上一次朝向角度
        self.turn_angle = 0.0  # 目标转向角度
        self.turn_direction = TURN_LEFT  # 转向方向
        self.goal_distance = random.uniform(1.0, 2.0)  # 目标行驶距离

        # 新增固定放置点位置（这里示例设置为x=3.0, y=3.0，可根据实际需求调整）
        self.drop_point = Pose()
        self.drop_point.position.x = -5.0
        self.drop_point.position.y = 0.0

        # 初始化传感器相关变量
        self.scan_triggered = [False] * 4  # 激光雷达触发标志
        self.items = ItemList()  # 检测到的物品列表
        self.item_held = False  # 是否持有物品

        # 初始化扫描相关变量
        self.scan_start_time = None  # 扫描开始时间
        self.scan_duration = 2.0  # 扫描持续时间（秒）

        # 声明和获取ROS参数
        self.declare_parameter('robot_id', 'robot1')
        self.robot_id = self.get_parameter('robot_id').value

        # 创建回调组
        client_callback_group = MutuallyExclusiveCallbackGroup()
        timer_callback_group = MutuallyExclusiveCallbackGroup()

        # 创建服务客户端
        self.pick_up_service = self.create_client(
            ItemRequest,
            '/pick_up_item',
            callback_group=client_callback_group
        )
        self.offload_service = self.create_client(
            ItemRequest,
            '/offload_item',
            callback_group=client_callback_group
        )

        # 创建订阅者
        self.item_subscriber = self.create_subscription(
            ItemList,
            'items',
            self.item_callback,
            10,
            callback_group=timer_callback_group
        )

        self.odom_subscriber = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10,
            callback_group=timer_callback_group
        )

        self.scan_subscriber = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            QoSPresetProfiles.SENSOR_DATA.value,
            callback_group=timer_callback_group
        )

        # 创建发布者
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )

        self.marker_publisher = self.create_publisher(
            StringWithPose,
            'marker_input',
            10,
            callback_group=timer_callback_group
        )

        # 创建控制循环定时器
        self.timer_period = 0.1  # 10Hz
        self.timer = self.create_timer(
            self.timer_period,
            self.control_loop,
            callback_group=timer_callback_group
        )

    def item_callback(self, msg):
        """处理物品检测消息的回调函数"""
        self.items = msg
        if len(self.items.data) > 0:
            self.get_logger().debug(f"检测到 {len(self.items.data)} 个物品")

    def odom_callback(self, msg):
        """处理里程计消息的回调函数"""
        self.pose = msg.pose.pose
        # 将四元数转换为欧拉角
        (roll, pitch, yaw) = euler_from_quaternion([
            self.pose.orientation.x,
            self.pose.orientation.y,
            self.pose.orientation.z,
            self.pose.orientation.w
        ])
        self.yaw = yaw

    def scan_callback(self, msg):
        """处理激光雷达扫描消息的回调函数"""
        # 将扫描数据分为四个区域
        front_ranges = msg.ranges[270:359] + msg.ranges[0:90]
        left_ranges = msg.ranges[90:180]
        back_ranges = msg.ranges[180:270]
        right_ranges = msg.ranges[270:360]

        # 更新障碍物检测标志
        self.scan_triggered[SCAN_FRONT] = min(front_ranges) < SCAN_THRESHOLD
        self.scan_triggered[SCAN_LEFT] = min(left_ranges) < SCAN_THRESHOLD
        self.scan_triggered[SCAN_BACK] = min(back_ranges) < SCAN_THRESHOLD
        self.scan_triggered[SCAN_RIGHT] = min(right_ranges) < SCAN_THRESHOLD

    def control_loop(self):
        """主控制循环 - 实现有限状态机"""
        # 发布当前状态到RViz
        marker_input = StringWithPose()
        marker_input.text = str(self.state)
        marker_input.pose = self.pose
        self.marker_publisher.publish(marker_input)

        # 状态机实现
        match self.state:
            case State.FORWARD:
                self._handle_forward_state()
            case State.TURNING:
                self._handle_turning_state()
            case State.COLLECTING:
                self._handle_collecting_state()
            case State.MOVING_TO_DROP_POINT:
                self._handle_moving_to_drop_point()
            case State.DROPPING:
                self._handle_dropping()

    def _handle_forward_state(self):
        """处理前进状态的逻辑"""
        if self.scan_triggered[SCAN_FRONT]:
            # 检测到前方障碍物，准备转向
            self._prepare_turn(150, 170)
            return

        if len(self.items.data) > 0:
            # 检测到物品，切换到收集状态
            self.state = State.COLLECTING
            return

        # 正常前进
        msg = Twist()
        msg.linear.x = LINEAR_VELOCITY
        self.cmd_vel_publisher.publish(msg)

        # 检查是否达到目标距离
        if self._check_distance_reached():
            self.state = State.COLLECTING
            self.get_logger().info("到达目标距离，准备收集物品")

    def _handle_turning_state(self):
        """处理转向状态的逻辑"""
        msg = Twist()
        msg.angular.z = self.turn_direction * ANGULAR_VELOCITY
        self.cmd_vel_publisher.publish(msg)

        # 检查是否完成转向
        yaw_difference = angles.normalize_angle(self.yaw - self.previous_yaw)
        if math.fabs(yaw_difference) >= math.radians(self.turn_angle):
            self._complete_turn()

    def _handle_collecting_state(self):
        """处理收集物品状态的逻辑"""
        if len(self.items.data) == 0:
            self._handle_scanning()
            return

        item = self.items.data[0]
        heading_error = item.x / 320.0
        distance = 32.4 * float(item.diameter) ** -0.75

        if distance <= 0.35:
            self._attempt_pickup()
        else:
            self._approach_item(distance, heading_error)

    def _handle_scanning(self):
        """处理扫描过程的逻辑"""
        current_time = self.get_clock().now()
        if self.scan_start_time is None:
            self.scan_start_time = current_time

        if (current_time - self.scan_start_time).nanoseconds < self.scan_duration * 1e9:
            msg = Twist()
            msg.angular.z = 0.3
            self.cmd_vel_publisher.publish(msg)
        else:
            self.scan_start_time = None
            self.state = State.FORWARD

    def _attempt_pickup(self):
        """尝试拾取物品的逻辑"""
        # 停止机器人
        stop_msg = Twist()
        self.cmd_vel_publisher.publish(stop_msg)

        # 创建并发送拾取请求
        rqt = ItemRequest.Request()
        rqt.robot_id = self.robot_id

        try:
            future = self.pick_up_service.call_async(rqt)
            self.executor.spin_until_future_complete(future)
            response = future.result()

            if response.success:
                self.get_logger().info('物品拾取成功')
                self.item_held = True
                self.state = State.MOVING_TO_DROP_POINT
                self.previous_pose = self.pose
                self.goal_distance = random.uniform(1.0, 2.0)
            else:
                self.get_logger().warn(f'物品拾取失败: {response.message}')
                self._handle_pickup_failure()

        except Exception as e:
            self.get_logger().error(f'拾取过程发生错误: {str(e)}')
            self.state = State.FORWARD

    def _handle_pickup_failure(self):
        """处理拾取失败的情况"""
        backup_msg = Twist()
        backup_msg.linear.x = -0.1
        self.cmd_vel_publisher.publish(backup_msg)
        self.state = State.FORWARD

    def _approach_item(self, distance, heading_error):
        """控制机器人接近物品"""
        msg = Twist()
        msg.linear.x = min(0.15, 0.2 * distance)
        msg.angular.z = 0.5 * heading_error
        self.cmd_vel_publisher.publish(msg)

    def _prepare_turn(self, min_angle, max_angle):
        """准备转向动作"""
        self.previous_yaw = self.yaw
        self.state = State.TURNING
        self.turn_angle = random.uniform(min_angle, max_angle)
        self.turn_direction = random.choice([TURN_LEFT, TURN_RIGHT])
        self.get_logger().info(
            f"开始转向 {self.turn_angle:.2f} 度，方向：" +
            ("左" if self.turn_direction == TURN_LEFT else "右")
        )

    def _complete_turn(self):
        """完成转向动作"""
        self.previous_pose = self.pose
        self.goal_distance = random.uniform(1.0, 2.0)
        self.state = State.FORWARD
        self.get_logger().info(f"转向完成，开始前进 {self.goal_distance:.2f} 米")

    def _check_distance_reached(self):
        """检查是否达到目标距离"""
        dx = self.pose.position.x - self.previous_pose.position.x
        dy = self.pose.position.y - self.previous_pose.position.y
        distance_travelled = math.sqrt(dx * dx + dy * dy)
        return distance_travelled >= self.goal_distance

    def _handle_moving_to_drop_point(self):
        """处理前往放置物品固定位置的逻辑"""
        # 计算当前位置与目标放置点位置的距离和角度偏差
        dx = self.drop_point.position.x - self.pose.position.x
        dy = self.drop_point.position.y - self.pose.position.y
        distance_to_drop_point = math.sqrt(dx * dx + dy * dy)
        target_yaw = math.atan2(dy, dx)
        yaw_difference = angles.normalize_angle(target_yaw - self.yaw)

        # 根据距离和角度偏差发布速度指令
        msg = Twist()
        msg.linear.x = min(0.1, 0.2 * distance_to_drop_point)
        msg.angular.z = 0.5 * yaw_difference

        self.cmd_vel_publisher.publish(msg)

        # 判断是否到达目标放置点附近（这里设定距离小于0.2米认为到达）
        if distance_to_drop_point < 0.2:
            self.state = State.DROPPING

    def _handle_dropping(self):
        """处理放置物品的逻辑"""
        # 停止机器人
        stop_msg = Twist()
        self.cmd_vel_publisher.publish(stop_msg)

        # 创建并发送卸载物品请求
        rqt = ItemRequest.Request()
        rqt.robot_id = self.robot_id

        try:
            future = self.offload_service.call_async(rqt)
            self.executor.spin_until_future_complete(future)
            response = future.result()

            if response.success:
                self.get_logger().info('物品放置成功')
                self.item_held = False
                self.state = State.FORWARD
                self.previous_pose = self.pose
                self.goal_distance = random.uniform(1.0, 2.0)
            else:
                self.get_logger().warn(f'物品放置失败: {response.message}')
            # 这里可以根据实际情况添加更多处理放置失败的逻辑

        except Exception as e:
            self.get_logger().error(f'放置过程发生错误: {str(e)}')

    def destroy_node(self):
        """清理并销毁节点"""
        # 停止机器人
        stop_msg = Twist()
        self.cmd_vel_publisher.publish(stop_msg)
        self.get_logger().info("正在停止机器人并清理资源")
        super().destroy_node()


def main(args=None):
    """主函数"""
    # 初始化ROS2
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)

    # 创建节点和执行器
    node = RobotController()
    executor = MultiThreadedExecutor()

    # 设置节点的执行器引用
    node.executor = executor

    # 将节点添加到执行器
    executor.add_node(node)

    try:
        # 运行执行器
        executor.spin()
    except KeyboardInterrupt:
        # 处理Ctrl+C中断
        node.get_logger().info("收到键盘中断信号，正在关闭节点...")
    except ExternalShutdownException:
        # 处理外部关闭信号
        node.get_logger().error("收到外部关闭信号")
        sys.exit(1)
    finally:
        # 清理资源
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

