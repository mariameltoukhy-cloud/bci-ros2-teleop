import time, numpy as np, joblib, rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from pylsl import StreamInlet, resolve_byprop
from features import bandpass, WIN

STEP = 40
ALPHA = 0.3
THRESH = 0.40
SPEED = 0.30
TURN = 0.35


class BCIDecoder(Node):
    def __init__(self):
        super().__init__("bci_decoder")
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.clf = joblib.load("csp_lda.joblib")

        eeg = resolve_byprop("type", "EEG", timeout=15)
        if not eeg:
            raise RuntimeError("No EEG stream. Is lsl_streamer.py running?")
        self.inlet = StreamInlet(eeg[0], max_buflen=10)
        self.inlet.open_stream()

        mk = resolve_byprop("type", "Markers", timeout=3)
        self.mk_inlet = StreamInlet(mk[0]) if mk else None
        if self.mk_inlet:
            self.mk_inlet.open_stream()
        self.truth = "-"

        self.buf = np.zeros((64, WIN))
        self.filled = 0
        self.n_new = 0
        self.p = np.array([1 / 3, 1 / 3, 1 / 3])
        self.log = open("log.csv", "w")
        self.log.write("t,truth,rest,left,right,turn\n")
        self.create_timer(0.02, self.tick)
        self.get_logger().info("Decoder ready, waiting for data...")

    def tick(self):
        if self.mk_inlet:
            m, _ = self.mk_inlet.pull_chunk(timeout=0.0)
            if m:
                self.truth = m[-1][0]

        chunk, ts = self.inlet.pull_chunk(timeout=0.0, max_samples=128)
        if not ts:
            return
        x = np.asarray(chunk, dtype=np.float64).T
        n = x.shape[1]
        self.buf = np.roll(self.buf, -n, axis=1)
        self.buf[:, -n:] = x
        self.filled += n
        self.n_new += n
        if self.filled < WIN or self.n_new < STEP:
            return
        self.n_new = 0

        p = self.clf.predict_proba(bandpass(self.buf)[None])[0]
        self.p = ALPHA * p + (1 - ALPHA) * self.p

        msg = Twist()
        msg.linear.x = SPEED
        if self.p[2] > THRESH:
            msg.angular.z = -TURN
        elif self.p[1] > THRESH:
            msg.angular.z = TURN
        self.log.write(f"{time.time():.3f},{self.truth},{self.p[0]:.3f},"
                       f"{self.p[1]:.3f},{self.p[2]:.3f},{msg.angular.z:.2f}\n")
        self.log.flush()
        self.pub.publish(msg)
        self.get_logger().info(
            f"truth={self.truth} rest={self.p[0]:.2f} "
            f"left={self.p[1]:.2f} right={self.p[2]:.2f} "
            f"turn={msg.angular.z:+.2f}")


def main():
    rclpy.init()
    node = BCIDecoder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.log.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
