import json
import time
import serial


class RobotInterface:
    def __init__(self, port="COM3", baudrate=115200, timeout=1.0, send_units="cm"):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.send_units = send_units
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(2)

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

    def _convert(self, value_mm):
        if self.send_units == "cm":
            return round(value_mm / 10.0, 3)
        return round(value_mm, 3)

    def send_json(self, payload):
        msg = json.dumps(payload)
        self.ser.write((msg + "\n").encode("utf-8"))
        print(f"[ROBOT SEND] {msg}")
        return self.read_response()

    def read_response(self):
        line = self.ser.readline().decode("utf-8", errors="ignore").strip()
        if line:
            print(f"[ROBOT RECV] {line}")
        return line

    def ready(self):
        return self.send_json({"READY": []})

    def home(self):
        return self.send_json({"HOME": []})

    def finish(self):
        return self.send_json({"FINISH": []})

    def stop(self):
        return self.send_json({"STOP": []})

    def move_xyz_mm(self, x_mm, y_mm, z_mm):
        return self.send_json({
            "MOVE": {
                "X": self._convert(x_mm),
                "Y": self._convert(y_mm),
                "Z": self._convert(z_mm),
            }
        })

    def get_state(self):
        response = self.ready()

        try:
            data = json.loads(response)
            if "OK" in data and len(data["OK"]) > 0:
                return data["OK"][0]
        except Exception:
            pass

        return None

    def wait_until_idle(self, poll_interval=0.2, timeout_s=30.0):
        start = time.time()

        while time.time() - start < timeout_s:
            state = self.get_state()

            if state == "IDLE":
                print("[ROBOT] State is IDLE")
                return True

            print(f"[ROBOT] Waiting, state={state}")
            time.sleep(poll_interval)

        raise TimeoutError("Robot did not become IDLE in time.")

    def move_xyz_mm_and_wait(self, x_mm, y_mm, z_mm, timeout_s=30.0):
        self.wait_until_idle()
        self.move_xyz_mm(x_mm, y_mm, z_mm)
        self.wait_until_idle(timeout_s=timeout_s)