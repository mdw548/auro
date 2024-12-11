# 导入系统模块
import sys
import time

# 导入 ROS2 的核心库
import rclpy  # ROS2 Python 客户端库
from rclpy.node import Node  # ROS2 节点基类
from rclpy.signals import SignalHandlerOptions  # 信号处理选项
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor  # 执行器和外部关闭异常
from rclpy.qos import QoSPresetProfiles  # QoS 预设配置
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup  # 回调组，用于控制并发执行

# 导入标准 ROS2 消息类型
from std_msgs.msg import Float32  # 标准浮点消息
from geometry_msgs.msg import Twist, Pose  # 几何消息，用于速度和位姿
from nav_msgs.msg import Odometry  # 里程计消息
from sensor_msgs.msg import LaserScan  # 激光扫描数据消息

# 导入自定义接口的消息和服务
from assessment_interfaces.msg import Item, ItemList  # 自定义消息类型
from auro_interfaces.msg import StringWithPose
from auro_interfaces.srv import ItemRequest # 自定义服务类型

# 导入数学工具
from tf_transformations import euler_from_quaternion  # 四元数到欧拉角转换
import angles  # 角度计算工具
from enum import Enum  # 枚举类型，用于定义状态
import random  # 随机数生成器
import math  # 数学函数库

# 定义机器人运动参数常量
LINEAR_VELOCITY = 0.2  # 机器人前进速度（米/秒）
ANGULAR_VELOCITY = 0.2  # 机器人转向速度（弧度/秒）

# 定义转向方向常量
TURN_LEFT = 1  # 向左转的角速度方向
TURN_RIGHT = -1  # 向右转的角速度方向

# 定义激光雷达参数
SCAN_THRESHOLD = 0.3  # 激光雷达检测距离阈值（米）
SCAN_WARN_THRESHOLD = 0.6  # 添加警告距离阈值，提前减速
SCAN_FRONT = 0  # 前方区域索引
SCAN_LEFT = 1  # 左侧区域索引
SCAN_BACK = 2  # 后方区域索引
SCAN_RIGHT = 3  # 右侧区域索引

# 定义机器人状态
class State(Enum):
    FORWARD = 0     # 向前行驶状态
    TURNING = 1     # 转向状态
    COLLECTING = 2  # 收集物品状态
    DELIVERING = 3  # 运送物品到区域状态

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

        #12.11 new
        self.initial_pose = None
        
        # 初始化传感器相关变量
        self.scan_triggered = [False] * 4  # 激光雷达触发标志
        self.items = ItemList()  # 检测到的物品列表
        self.item_held = False  # 是否持有物品
        self.held_item_color = None
        
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

        # 定义区域位置
        self.zones = {
            'GREEN': {'x': 0.07, 'y': 2.5},      # 右上角
            'RED': {'x': 0.02, 'y': 2.35},     # 右下角
            'PURPLE': {'x': -3.42, 'y': -2.46}, # 左下角
            'BLUE': {'x': -3.42, 'y': 2.5}       # 左上角
        }

    

    def item_callback(self, msg):
        """处理物品检测消息的回调函数"""
        self.items = msg
        if len(self.items.data) > 0:
            self.get_logger().debug(f"检测到 {len(self.items.data)} 个物品")

    def odom_callback(self, msg):

        # 记录初始位置
        if self.initial_pose is None:
            self.initial_pose = msg.pose.pose
            self.get_logger().info(f"初始位置记录为: ({self.initial_pose.position.x:.2f}, {self.initial_pose.position.y:.2f})")

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

        
        # 过滤掉无效的测量值
        valid_front = [r for r in front_ranges if not math.isinf(r) and not math.isnan(r)]
        valid_left = [r for r in left_ranges if not math.isinf(r) and not math.isnan(r)]
        valid_right = [r for r in right_ranges if not math.isinf(r) and not math.isnan(r)]
        valid_back = [r for r in back_ranges if not math.isinf(r) and not math.isnan(r)]


        if valid_front:
            min_front = min(valid_front)

            # 如果当前检测到有颜色物品(即 items 列表不为空)
            if len(self.items.data) > 0:
                min_front = float('inf')
            
            self.scan_triggered[SCAN_FRONT] = min_front < SCAN_THRESHOLD
            if min_front < SCAN_WARN_THRESHOLD:
                self.front_distance = min_front
            else:
                self.front_distance = float('inf')
        
        if valid_left:
            self.scan_triggered[SCAN_LEFT] = min(valid_left) < SCAN_THRESHOLD
        if valid_right:
            self.scan_triggered[SCAN_RIGHT] = min(valid_right) < SCAN_THRESHOLD
        if valid_back:
            self.scan_triggered[SCAN_BACK] = min(valid_back) < SCAN_THRESHOLD

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
            case State.DELIVERING:
                self._handle_delivering_state()
            case State.RETURNING:
                self._handle_returning_state()


    def _handle_forward_state(self):
        """处理前进状态的逻辑"""
        if self.scan_triggered[SCAN_FRONT]:
            # 检测到前方障碍物，准备转向
            self._prepare_turn(150, 170)
            return

        if self.item_held:
            self.state = State.DELIVERING
            self.get_logger().info(f"准备将{self.held_item_color}颜色的物品运送到对应区域")
            return

        # 检测可见的物品，计算距离并排序
        if len(self.items.data) > 0 and not self.item_held:
            items_with_distance = []
            for item in self.items.data:
                distance = 32.4 * float(item.diameter) ** -0.75
                items_with_distance.append({
                    'item': item,
                    'distance': distance,
                    'color': item.colour
                })
            
            items_with_distance.sort(key=lambda x: x['distance'])
            nearest_item = items_with_distance[0]
            
            self.get_logger().info(
                f"前进过程中发现物品:\n"
                f"最近物品: {nearest_item['color']} 颜色, 距离: {nearest_item['distance']:.2f}米"
            )
            
            # 如果有物品比较近，切换到收集状态
            if nearest_item['distance'] < 2.0:  # 可以调整这个阈值
                self.state = State.COLLECTING
                return

        # 正常前进
        msg = Twist()
        msg.linear.x = LINEAR_VELOCITY
        self.cmd_vel_publisher.publish(msg)

        # 检查是否达到目标距离
        if self._check_distance_reached():
            self.state = State.COLLECTING
            self.get_logger().info("到达目标距离，准备搜索物品")



    def _handle_turning_state(self):
        """处理转向状态的逻辑"""
        msg = Twist()
        msg.angular.z = self.turn_direction * ANGULAR_VELOCITY
        self.cmd_vel_publisher.publish(msg)

        # 检查是否完成转向
        yaw_difference = angles.normalize_angle(self.yaw - self.previous_yaw)
        if math.fabs(yaw_difference) >= math.radians(self.turn_angle):
            self._complete_turn()

    

    #NEW
    def _handle_collecting_state(self):
        """处理收集物品状态的逻辑"""
        if len(self.items.data) == 0:
            self._handle_scanning()
            return

        if self._check_and_avoid_obstacles():
            return

        # 计算所有可见物品的距离，并按距离排序
        items_with_distance = []
        for item in self.items.data:
            distance = 32.4 * float(item.diameter) ** -0.75  # 使用已有公式计算距离
            items_with_distance.append({
                'item': item,
                'distance': distance,
                'color': item.colour
            })
        
        # 按距离排序
        items_with_distance.sort(key=lambda x: x['distance'])
        
        # 输出所有可见物品的信息
        for item_info in items_with_distance:
            self.get_logger().info(
                f"检测到 {item_info['color']} 颜色物品，距离: {item_info['distance']:.2f}米"
            )

        # 选择最近的物品
        nearest_item = items_with_distance[0]
        item = nearest_item['item']
        distance = nearest_item['distance']
        
        self.get_logger().info(
            f"选择最近的物品进行收集:\n"
            f"颜色: {item.colour}\n"
            f"距离: {distance:.2f}米"
        )

        heading_error = item.x / 320.0

        if distance <= 0.35:
            self._attempt_pickup()
        else:
            self._approach_item(distance, heading_error)


    def _handle_scanning(self):

        if self._check_and_avoid_obstacles():
            return
        

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

        # 获取当前目标物品的颜色
        target_color = self.items.data[0].colour.upper() if len(self.items.data) > 0 else "UNKNOWN"

        # 创建并发送拾取请求
        rqt = ItemRequest.Request()
        rqt.robot_id = self.robot_id

        try:
            future = self.pick_up_service.call_async(rqt)
            self.executor.spin_until_future_complete(future)
            response = future.result()

            if response.success:
                self.get_logger().info(f'{target_color}颜色的物品拾取成功')
                self.item_held = True
                self.held_item_color = target_color
                self.state = State.FORWARD
                self.previous_pose = self.pose
                self.goal_distance = random.uniform(1.0, 2.0)
            else:
                self.get_logger().warn(f'{target_color}颜色的物品拾取失败: {response.message}')
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
    
        # 根据前方障碍物距离调整速度
        if hasattr(self, 'front_distance') and self.front_distance < SCAN_WARN_THRESHOLD:
            # 根据距离逐渐减速
            speed_factor = max(0.3, self.front_distance / SCAN_WARN_THRESHOLD)
            msg.linear.x = min(0.15, 0.2 * distance) * speed_factor
            self.get_logger().info(f'检测到前方障碍物，减速至 {speed_factor:.2f}')
        else:
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

    def _handle_delivering_state(self):
        """处理运送物品到区域的逻辑"""
        # 1. 基础检查
        if not self.item_held:
            self.get_logger().debug("未持有物品，切换到前进状态")
            self.state = State.FORWARD
            return

        # 2. 获取目标区域
        target_zone = self.zones.get(self.held_item_color)
        if target_zone is None:
            self.get_logger().error(f"未找到{self.held_item_color}颜色对应的区域")
            self.state = State.FORWARD
            return

        # 3. 计算位置信息
        dx = target_zone['x'] - self.pose.position.x
        dy = target_zone['y'] - self.pose.position.y
        distance = math.sqrt(dx * dx + dy * dy)
        target_angle = math.atan2(dy, dx)
        angle_diff = angles.normalize_angle(target_angle - self.yaw)

        # 4. 输出详细的导航状态日志
        self.get_logger().info(
            f"导航状态:\n"
            f"机器人当前位置: ({self.pose.position.x:.2f}, {self.pose.position.y:.2f})\n"
            f"目标区域位置: ({target_zone['x']:.2f}, {target_zone['y']:.2f})\n"
            f"距离目标: {distance:.2f}米\n"
            f"角度差: {math.degrees(angle_diff):.2f}度"
        )

        # 5. 避障检查
        if self._check_and_avoid_obstacles():
            return

        # 6. 目标达成检查
        if distance < 0.5:
            self.get_logger().info("已到达目标区域附近，尝试放下物品")
            self._attempt_offload()
            return

        # 7. 运动控制
        msg = Twist()
        
        # 角度调整
        if abs(angle_diff) > 0.1:  # 约5.7度
            # 角度调整时停止前进，专注于转向
            msg.angular.z = 0.3 if angle_diff > 0 else -0.3
            self.get_logger().debug(f"调整朝向，当前角度差: {math.degrees(angle_diff):.2f}度")
        else:
            # 前进速度控制
            base_speed = min(0.2, distance * 0.5)
            
            # 障碍物检测减速
            if hasattr(self, 'front_distance') and self.front_distance < SCAN_WARN_THRESHOLD:
                speed_factor = max(0.3, self.front_distance / SCAN_WARN_THRESHOLD)
                msg.linear.x = base_speed * speed_factor
                self.get_logger().info(f'检测到障碍物，减速至 {speed_factor:.2f}')
            else:
                msg.linear.x = base_speed
                
            # 微调角度
            msg.angular.z = angle_diff * 0.5  # 添加比例因子使转向更平滑

        # 8. 发布控制命令
        try:
            self.cmd_vel_publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f"发布速度命令时出错: {str(e)}")
    
    def _attempt_offload(self):
        """尝试在区域放下物品"""
        # 停止机器人
        stop_msg = Twist()
        self.cmd_vel_publisher.publish(stop_msg)

        # 创建并发送卸载请求
        rqt = ItemRequest.Request()
        rqt.robot_id = self.robot_id

        try:
            future = self.offload_service.call_async(rqt)
            self.executor.spin_until_future_complete(future)
            response = future.result()

            if response.success:
                self.get_logger().info(f'成功在{self.held_item_color}区域放下物品')
                self.item_held = False
                self.held_item_color = None
                
                # 首先后退一小段距离，避免与墙壁太近
                backup_msg = Twist()
                backup_msg.linear.x = -0.15
                self.cmd_vel_publisher.publish(backup_msg)
                time.sleep(1.0)  # 后退1秒
                
                # 先转向远离墙壁
                self.state = State.TURNING
                self.previous_yaw = self.yaw
                self.turn_angle = random.uniform(150, 180)  # 大角度转向
                self.turn_direction = random.choice([TURN_LEFT, TURN_RIGHT])
                
                # 记录新的位置作为起点
                self.previous_pose = self.pose
                # 设置较短的前进距离
                self.goal_distance = random.uniform(0.5, 1.0)
                
                self.get_logger().info('成功放下物品，开始转向寻找新的物品')
            else:
                self.get_logger().warn(f'物品放置失败: {response.message}')
                # 后退一点并重试
                backup_msg = Twist()
                backup_msg.linear.x = -0.1
                self.cmd_vel_publisher.publish(backup_msg)

        except Exception as e:
            self.get_logger().error(f'放置过程发生错误: {str(e)}')
            self.state = State.TURNING  # 这里也改为TURNING
    




    def destroy_node(self):
        """清理并销毁节点"""
        try:
            # 停止机器人
            stop_msg = Twist()
            self.cmd_vel_publisher.publish(stop_msg)
            self.get_logger().info("正在停止机器人并清理资源")
        finally:
            super().destroy_node()

    # 12.7
    def _check_and_avoid_obstacles(self):
        """优化后的避障检查方法，使用更智能的避障策略"""
        # 检查四个方向的障碍物情况
        front_blocked = self.scan_triggered[SCAN_FRONT]
        left_blocked = self.scan_triggered[SCAN_LEFT]
        right_blocked = self.scan_triggered[SCAN_RIGHT]
        back_blocked = self.scan_triggered[SCAN_BACK]

        # 如果没有障碍物，直接返回
        if not any([front_blocked, left_blocked, right_blocked, back_blocked]):
            return False

        # 获取当前目标方向（如果在运送状态）
        target_angle = None
        if self.state == State.DELIVERING and self.held_item_color:
            target_zone = self.zones.get(self.held_item_color)
            if target_zone:
                dx = target_zone['x'] - self.pose.position.x
                dy = target_zone['y'] - self.pose.position.y
                target_angle = math.atan2(dy, dx)
        elif self.state == State.RETURNING:
            target_pos = self.search_positions[self.current_search_index]
            dx = target_pos['x'] - self.pose.position.x
            dy = target_pos['y'] - self.pose.position.y
            target_angle = math.atan2(dy, dx)

        # 计算最佳避障策略
        if front_blocked:
            # 如果前方被阻挡
            if not left_blocked and not right_blocked:
                # 两侧都可以转向时，选择最优方向
                if target_angle is not None:
                    # 计算左转和右转后与目标方向的角度差
                    angle_diff = angles.normalize_angle(target_angle - self.yaw)
                    if angle_diff > 0:
                        self._smooth_turn(TURN_LEFT)
                    else:
                        self._smooth_turn(TURN_RIGHT)
                else:
                    # 没有特定目标时，选择障碍物较少的一侧
                    self._smooth_turn(TURN_LEFT if not left_blocked else TURN_RIGHT)
            elif not left_blocked:
                self._smooth_turn(TURN_LEFT)
            elif not right_blocked:
                self._smooth_turn(TURN_RIGHT)
            else:
                # 三个方向都被阻挡，执行后退和大角度转向
                self._emergency_maneuver()

        return True
    
    def _smooth_turn(self, direction, base_speed=0.2):
        """执行平滑转向"""
        msg = Twist()
        # 根据障碍物距离动态调整转向速度
        if hasattr(self, 'front_distance'):
            turn_speed = min(0.5, max(0.2, 1.0 - self.front_distance / SCAN_WARN_THRESHOLD))
        else:
            turn_speed = 0.3
        
        msg.angular.z = turn_speed * direction
        msg.linear.x = base_speed * (1.0 - abs(msg.angular.z))  # 转向时适当降低前进速度
        self.cmd_vel_publisher.publish(msg)
        self.get_logger().info(f"执行平滑转向，方向: {'左' if direction == TURN_LEFT else '右'}, 速度: {turn_speed:.2f}")

    def _emergency_maneuver(self):
        """处理紧急情况的避障动作"""
        self.get_logger().warn("执行紧急避障操作")
        
        # 首先尝试后退
        backup_msg = Twist()
        backup_msg.linear.x = -0.15
        self.cmd_vel_publisher.publish(backup_msg)
        
        # 等待短暂时间
        time.sleep(0.5)
        
        # 检查周围环境是否已经安全
        if not any(self.scan_triggered):
            self.get_logger().info("环境已安全，恢复正常运行")
            return
        
        # 如果仍然不安全，执行大角度转向
        self._prepare_turn(150, 180, force_direction=random.choice([TURN_LEFT, TURN_RIGHT]))

    def _prepare_turn(self, min_angle, max_angle, force_direction=None):
        """优化后的转向准备函数"""
        self.previous_yaw = self.yaw
        self.state = State.TURNING
        
        # 根据当前状态调整转向角度
        if self.state == State.DELIVERING or self.state == State.RETURNING:
            # 在特定任务状态下，使用较小的转向角度以保持大致方向
            self.turn_angle = random.uniform(min_angle/2, max_angle/2)
        else:
            self.turn_angle = random.uniform(min_angle, max_angle)
        
        if force_direction is not None:
            self.turn_direction = force_direction
        else:
            # 增加转向方向的智能选择
            left_preference = 0
            right_preference = 0
            
            # 考虑障碍物分布
            if self.scan_triggered[SCAN_LEFT]:
                right_preference += 1
            if self.scan_triggered[SCAN_RIGHT]:
                left_preference += 1
                
            # 根据偏好选择方向
            self.turn_direction = TURN_LEFT if left_preference <= right_preference else TURN_RIGHT
        
        self.get_logger().info(
            f"准备转向 {self.turn_angle:.2f} 度，方向：" +
            ("左" if self.turn_direction == TURN_LEFT else "右")
        )

def main(args=None):
    """主函数"""
    # 初始化ROS2
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)

    # 创建节点和执行器
    node = RobotController()
    executor = MultiThreadedExecutor()
    
    # 设置节点的执行器 引用
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
