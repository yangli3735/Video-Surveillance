"""
modules/gpio/gpio_controller.py
=================================
Responsibility:
  - Abstract GPIO operations (buzzer pulse, LED, relay, PIR sensor read)
  - Use Jetson.GPIO library (NVIDIA's RPi.GPIO-compatible package)
  - Provide a safe no-op stub when gpio.enabled=false (dev machine / headless CI)
  - Expose on_event() so it can be registered directly as an EventHandler callback

Jetson.GPIO installation:
    pip install Jetson.GPIO
    sudo groupadd -f -r gpio
    sudo usermod -a -G gpio $USER
    sudo cp /opt/nvidia/jetson-gpio/etc/99-gpio.rules /etc/udev/rules.d/
    sudo udevadm control --reload-rules && sudo udevadm trigger
    (reboot or re-login required)

Pin numbering:
  - BCM mode is used (same as Raspberry Pi convention)
  - Adjust buzzer_pin / motion_sensor_pin in config/settings.yaml

Future extensions:
  - Relay for lights / sirens
  - I2C / SPI sensors (temperature, smoke)
  - PWM buzzer tones
"""
import threading
import time
from typing import TYPE_CHECKING

from config import cfg
from utils.logger import get_logger

if TYPE_CHECKING:
    from modules.event.event_handler import Event

log = get_logger(__name__)


class GPIOController:
    """
    GPIO abstraction for buzzer and sensors on Jetson Orin Nano.

    Usage:
        gpio = GPIOController()
        gpio.setup()

        # Wire as event callback:
        event_handler.register_callback(gpio.on_event)

        # Clean up on shutdown:
        gpio.cleanup()
    """

    def __init__(self):
        gpio_cfg = cfg["gpio"]
        self._enabled: bool = gpio_cfg.get("enabled", False)
        self._buzzer_pin: int = gpio_cfg.get("buzzer_pin", 18)
        self._buzzer_duration_ms: int = gpio_cfg.get("buzzer_duration_ms", 500)
        self._motion_pin: int = gpio_cfg.get("motion_sensor_pin", 24)

        self._gpio = None   # set after setup()
        self._buzzer_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(self) -> None:
        if not self._enabled:
            log.info("GPIOController: disabled in config — running in stub mode")
            return
        try:
            import Jetson.GPIO as GPIO  # type: ignore
            self._gpio = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self._buzzer_pin, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(self._motion_pin, GPIO.IN)
            log.info(
                "GPIOController ready (buzzer=BCM%d, motion=BCM%d)",
                self._buzzer_pin, self._motion_pin,
            )
        except ImportError:
            log.warning(
                "Jetson.GPIO not installed — GPIO will run in stub mode. "
                "Install with: pip install Jetson.GPIO"
            )
            self._enabled = False
        except Exception as exc:
            log.error("GPIOController setup failed: %s", exc)
            self._enabled = False

    def cleanup(self) -> None:
        if self._gpio is not None:
            self._gpio.cleanup()
            log.info("GPIOController cleanup done")

    def on_event(self, event: "Event") -> None:
        """EventHandler callback — triggers buzzer in a background thread."""
        log.debug("GPIOController: event received (%s) — triggering buzzer", event.trigger_class)
        t = threading.Thread(
            target=self._buzz,
            args=(self._buzzer_duration_ms / 1000.0,),
            daemon=True,
        )
        t.start()

    def read_motion(self) -> bool:
        """Return True if the PIR sensor detects motion."""
        if not self._enabled or self._gpio is None:
            return False
        try:
            return bool(self._gpio.input(self._motion_pin))
        except Exception as exc:
            log.error("PIR read error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _buzz(self, duration: float) -> None:
        if not self._enabled or self._gpio is None:
            log.debug("GPIOController stub: buzz %.2fs", duration)
            return
        with self._buzzer_lock:
            try:
                self._gpio.output(self._buzzer_pin, self._gpio.HIGH)
                time.sleep(duration)
                self._gpio.output(self._buzzer_pin, self._gpio.LOW)
            except Exception as exc:
                log.error("Buzzer error: %s", exc)
