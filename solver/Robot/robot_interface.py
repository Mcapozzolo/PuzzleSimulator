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
            return round(float(value_mm) / 10.0, 3)
        return round(float(value_mm), 3)

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

    def clamp_c(self, c_deg):
        """
        Firmware akzeptiert C nur sicher im Bereich 2 bis 178.
        Werte nahe 1/179 können durch Rundung/Integer-Konvertierung fehlschlagen.
        """
        c_deg = float(c_deg)

        if c_deg < 2.0:
            return 2

        if c_deg > 178.0:
            return 178

        return int(round(c_deg))

    def move_xyzc_mm(self, x_mm, y_mm, z_mm, c_deg=90.0, pump=None):
        """
        Sendet MOVE an die Firmware.

        Wenn pump=None:
            PUMP wird nicht mitgeschickt.

        Wenn pump=True/False:
            PUMP wird explizit mitgeschickt.
        """
        c_deg = self.clamp_c(c_deg)

        move = {
            "X": self._convert(x_mm),
            "Y": self._convert(y_mm),
            "Z": self._convert(z_mm),
            "C": c_deg,
        }

        if pump is not None:
            move["PUMP"] = bool(pump)

        return self.send_json({
            "MOVE": move
        })

    def wait_for_message_contains(self, text, timeout_s=3.0):
        """
        Wartet auf eine bestimmte LOG-/Antwortzeile der Firmware.
        Gibt True zurück, wenn die Zeile gefunden wurde.
        """
        start = time.time()

        while time.time() - start < timeout_s:
            line = self.ser.readline().decode("utf-8", errors="ignore").strip()

            if not line:
                continue

            print(f"[ROBOT RECV] {line}")

            if text in line:
                return True

        print(f"[ROBOT WARN] Timeout beim Warten auf: {text}")
        return False

    def move_xyzc_mm_and_wait(self, x_mm, y_mm, z_mm, c_deg=90.0, pump=None, timeout_s=30.0):
        self.wait_until_idle()

        response = self.move_xyzc_mm(
            x_mm,
            y_mm,
            z_mm,
            c_deg,
            pump=pump,
        )

        if response and "NOTOK" in response and "Controller is not IDLE" not in response:
            raise RuntimeError(f"Robot rejected MOVE: {response}")

        time.sleep(0.15)
        self.wait_until_idle(timeout_s=timeout_s)

        return response

    def get_state(self):
        response = self.ready()

        try:
            data = json.loads(response)
            if "OK" in data and len(data["OK"]) > 0:
                return data["OK"][0]
        except Exception:
            pass

        return None

    def wait_until_idle(self, poll_interval=0.3, timeout_s=30.0, allow_homing=False):
        start = time.time()

        while time.time() - start < timeout_s:
            response = self.ready()

            if not response:
                time.sleep(poll_interval)
                continue

            try:
                data = json.loads(response)

                if "OK" in data:
                    ok = data["OK"]

                    if len(ok) == 0:
                        time.sleep(poll_interval)
                        continue

                    state = ok[0]

                    if state == "IDLE":
                        print("[ROBOT] State is IDLE")
                        return True

                    if state == "MOVING":
                        print("[ROBOT] Waiting, state=MOVING")
                        time.sleep(poll_interval)
                        continue

                    if state == "HOMING":
                        if allow_homing:
                            print("[ROBOT] Waiting, state=HOMING")
                            time.sleep(poll_interval)
                            continue

                        raise RuntimeError(
                            "Roboter ist noch im HOMING-State, obwohl Homing deaktiviert ist. "
                            "Bitte Arduino/Controller resetten oder STOP senden."
                        )

                if "LOG" in data:
                    print(f"[ROBOT] LOG beim Warten: {data['LOG']}")
                    time.sleep(poll_interval)
                    continue

                if "NOTOK" in data:
                    print(f"[ROBOT] Waiting, response={response}")
                    time.sleep(poll_interval)
                    continue

            except RuntimeError:
                raise

            except Exception:
                print(f"[ROBOT] Ignoriere Antwort beim Warten: {response}")
                time.sleep(poll_interval)
                continue

            time.sleep(poll_interval)

        raise TimeoutError(
            f"Robot did not become IDLE in time after {timeout_s:.1f}s."
        )

    def suction_on(self):
        raise RuntimeError("Nicht mehr verwenden. PUMP muss im MOVE-Command gesendet werden.")

    def suction_off(self):
        raise RuntimeError("Nicht mehr verwenden. PUMP muss im MOVE-Command gesendet werden.")

    def rotate_c(self, *args, **kwargs):
        raise RuntimeError("Nicht mehr separat verwenden. C muss im vollständigen MOVE-Command gesendet werden.")