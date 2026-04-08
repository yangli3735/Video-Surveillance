"""
modules/event/event_handler.py
================================
Responsibility:
  - Consume DetectionResult objects produced by the YOLO detector
  - Apply business rules to decide whether an event has occurred:
      * Detected class is in the watch-list
      * Confidence meets the threshold
      * Cooldown period between repeated events has passed
  - Fire callbacks registered by other modules (recorder, GPIO, cloud, etc.)
  - Runs in the MAIN thread (lightweight; callbacks dispatch to other threads)

Extending this module later:
  - Add a rule engine (e.g., detect person AND motion sensor active)
  - Add an action-recognition hook
  - Add a database write callback
  - Add a cloud notification callback
"""
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from config import cfg
from modules.inference.yolo_detector import DetectionResult, Detection
from utils.logger import get_logger

log = get_logger(__name__)


# ------------------------------------------------------------------
# Event data model
# ------------------------------------------------------------------

@dataclass
class Event:
    trigger_class: str
    confidence: float
    detection: Detection
    result: DetectionResult
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        return (
            f"[EVENT] {self.trigger_class} "
            f"(conf={self.confidence:.2f}) "
            f"at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}"
        )


# ------------------------------------------------------------------
# Event handler
# ------------------------------------------------------------------

class EventHandler:
    """
    Stateless rule evaluator + callback dispatcher.

    Usage:
        handler = EventHandler()
        handler.register_callback(my_function)   # fn(event: Event) -> None

        # In your main loop (or called from inference thread):
        handler.process(detection_result)
    """

    def __init__(self):
        ev_cfg = cfg["event"]
        self._watch: List[str] = [c.lower() for c in ev_cfg.get("watch_classes", [])]
        self._min_conf: float = ev_cfg.get("min_confidence", 0.60)
        self._cooldown: float = ev_cfg.get("cooldown_seconds", 5.0)

        # Track last event time per class for cooldown
        self._last_event: Dict[str, float] = {}
        self._callbacks: List[Callable[[Event], None]] = []

        log.info(
            "EventHandler ready — watching: %s | cooldown: %.1fs",
            self._watch, self._cooldown,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_callback(self, fn: Callable[[Event], None]) -> None:
        """Register a function to be called when an event fires."""
        self._callbacks.append(fn)
        log.debug("EventHandler: registered callback %s", fn.__name__)

    def process(self, result: DetectionResult) -> Optional[Event]:
        """
        Evaluate detections and fire callbacks if event conditions are met.
        Returns the fired Event (or None).
        """
        for det in result.detections:
            if det.class_name.lower() not in self._watch:
                continue
            if det.confidence < self._min_conf:
                continue

            now = time.time()
            last = self._last_event.get(det.class_name, 0.0)
            if now - last < self._cooldown:
                continue  # Still in cooldown window

            # ---- Event fires ----
            self._last_event[det.class_name] = now
            event = Event(
                trigger_class=det.class_name,
                confidence=det.confidence,
                detection=det,
                result=result,
            )
            log.info(str(event))
            self._dispatch(event)
            return event

        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _dispatch(self, event: Event) -> None:
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception as exc:
                log.error("Callback %s raised an error: %s", cb.__name__, exc)
