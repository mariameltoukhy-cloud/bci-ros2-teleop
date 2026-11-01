import time, rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

R, L = 0.0975, 0.381      # wheel radius, wheel separation (Pioneer P3DX)
TIMEOUT = 0.5             # stop if no command for this long (s)

class Bridge(Node):
    def __init__(self):
        super().__init__("coppelia_bridge")
        self.sim = RemoteAPIClient().require("sim")
        self.lm = self.sim.getObject("/PioneerP3DX/leftMotor")
        self.rm = self.sim.getObject("/PioneerP3DX/rightMotor")
        self.last = time.time()
        self.create_subscription(Twist, "/cmd_vel", self.cb, 10)
        self.create_timer(0.1, self.watchdog)
        self.get_logger().info("Bridge ready, listening on /cmd_vel")

    def drive(self, v, w):
        self.sim.setJointTargetVelocity(self.lm, (v - w * L / 2) / R)
        self.sim.setJointTargetVelocity(self.rm, (v + w * L / 2) / R)

    def cb(self, m):
        self.last = time.time()
        self.drive(m.linear.x, m.angular.z)

    def watchdog(self):
        if time.time() - self.last > TIMEOUT:
            self.drive(0.0, 0.0)

def main():
    rclpy.init()
    node = Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.drive(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
