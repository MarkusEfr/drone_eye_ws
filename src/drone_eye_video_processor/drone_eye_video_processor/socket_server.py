#!/usr/bin/env python3
"""
WS Video Server (aiohttp) + ROS2 publisher

Features:
- Serves a static client from ./static (index.html etc.)
- WebSocket endpoint at /ws accepting:
    - binary messages (JPEG/PNG bytes)
    - text messages containing base64 data URLs (data:image/jpeg;base64,...)
- Publishes decoded frames to ROS2 topic "video_frames" (sensor_msgs/Image)
- CORS-friendly and optional origin check
- Graceful shutdown of aiohttp & rclpy executor
"""

from pathlib import Path
import asyncio
import base64
import logging
import signal

import cv2
import numpy as np
from aiohttp import web, WSMsgType
import aiohttp_cors

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from drone_eye_msgs.msg import BoundingBoxes

_LOG = logging.getLogger("video_ws_server")
logging.basicConfig(level=logging.INFO)


# Allowed origin(s) for WebSocket connections. Set to None to accept any origin.
ALLOWED_ORIGINS = None  # e.g. {"http://localhost:8000", "http://myhost:8000"}


class SocketServer(Node):
    def __init__(self, topic_name: str = "/camera/image_raw", qos_depth: int = 10):
        super().__init__("video_ws_server")
        self.publisher = self.create_publisher(Image, topic_name, qos_depth)
        self.bridge = CvBridge()
        self.active_ws = None  # store the connected websocket

        # Subscribe to detector outputs
        self.create_subscription(
            Image, "/drone_eye/video_results", self.annotated_callback, 10
        )
        self.create_subscription(
            BoundingBoxes, "/drone_eye/detections", self.detections_callback, 10
        )

        self.get_logger().info(f"SocketServer publishing to: {topic_name}")

    def publish_frame(self, frame: np.ndarray):
        """Publish a BGR OpenCV frame to ROS2 as sensor_msgs/Image."""
        try:
            ros_img = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.publisher.publish(ros_img)
        except Exception as e:
            self.get_logger().error(f"Failed to publish frame: {e}")

    def annotated_callback(self, msg: Image):
        """Send annotated video frame back to the client."""
        if self.active_ws is None:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            _, jpeg_bytes = cv2.imencode(".jpg", frame)
            b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
            payload = {
                "type": "annotated_frame",
                "data": f"data:image/jpeg;base64,{b64}",
            }
            asyncio.create_task(self.active_ws.send_json(payload))
        except Exception as e:
            self.get_logger().error(f"Error sending annotated frame: {e}")

    def detections_callback(self, msg: BoundingBoxes):
        """Send detection metadata back to the client."""
        if self.active_ws is None:
            return
        try:
            boxes = []
            for box in msg.boxes:
                boxes.append(
                    {
                        "label": box.label,
                        "confidence": box.probability,
                        "bbox": [box.xmin, box.ymin, box.xmax, box.ymax],
                    }
                )
            payload = {"type": "detections", "data": boxes}
            asyncio.create_task(self.active_ws.send_json(payload))
        except Exception as e:
            self.get_logger().error(f"Error sending detection data: {e}")


def annotated_callback(self, msg: Image):
    """Send annotated video frame back to the client."""
    if self.active_ws is None:
        return
    try:
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        _, jpeg_np = cv2.imencode(".jpg", frame)
        jpeg_bytes = jpeg_np.tobytes()  # convert to bytes
        b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
        payload = {"type": "annotated_frame", "data": f"data:image/jpeg;base64,{b64}"}
        asyncio.create_task(self.active_ws.send_json(payload))
    except Exception as e:
        self.get_logger().error(f"Error sending annotated frame: {e}")


async def websocket_handler(self, request: web.Request) -> web.StreamResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    self.active_ws = ws
    self.get_logger().info("WebSocket client connected.")

    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                np_arr = np.frombuffer(msg.data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    self.publish_frame(frame)

            elif msg.type == WSMsgType.TEXT:
                if msg.data.startswith("data:image"):
                    try:
                        b64 = msg.data.split(",", 1)[1]
                        img_data = base64.b64decode(b64)
                        np_arr = np.frombuffer(img_data, np.uint8)
                        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            self.publish_frame(frame)
                    except Exception as e:
                        self.get_logger().error(f"Base64 decode error: {e}")

            elif msg.type == WSMsgType.ERROR:
                self.get_logger().error(f"WebSocket error: {ws.exception()}")

    finally:
        self.get_logger().info("WebSocket connection closed.")
        self.active_ws = None

    return ws


async def upload_handler(self, request):
    reader = await request.multipart()
    field = await reader.next()
    if field.name == "file":
        filename = field.filename
        data = await field.read()
        np_arr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is not None:
            ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.publisher.publish(ros_image)
            return web.Response(text=f"Uploaded and published: {filename}")
    return web.Response(status=400, text="Invalid file")


async def init_app(node: SocketServer, host: str = "0.0.0.0", port: int = 8000):
    """Create aiohttp app with static serving and WS route, plus CORS."""
    app = web.Application()

    # Add WS route
    app.router.add_get("/ws", node.websocket_handler)

    # Static folder (./static next to this file)
    pkg_dir = Path(__file__).resolve().parent
    static_dir = pkg_dir / "static"
    if not static_dir.exists():
        _LOG.warning("Static dir %s does not exist; creating empty folder.", static_dir)
        static_dir.mkdir(parents=True, exist_ok=True)
    app.router.add_static("/", str(static_dir), show_index=True)

    # Attach ROS node to app for handlers
    app["ros_node"] = node

    # Configure CORS to allow browser clients (adjust origins if needed)
    cors = aiohttp_cors.setup(
        app,
        defaults={
            origin: aiohttp_cors.ResourceOptions(
                allow_credentials=True, expose_headers="*", allow_headers="*"
            )
            for origin in (["*"] if ALLOWED_ORIGINS is None else ALLOWED_ORIGINS)
        },
    )
    # apply cors to all routes (including static)
    for route in list(app.router.routes()):
        try:
            cors.add(route)
        except Exception:
            pass

    return app


def run(loop: asyncio.AbstractEventLoop, node: SocketServer, host="0.0.0.0", port=8000):
    """Start aiohttp server and run until cancelled. Runs in the given asyncio loop."""
    # Create app and runner
    app = loop.run_until_complete(init_app(node, host=host, port=port))
    runner = web.AppRunner(app)
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, host, port)
    loop.run_until_complete(site.start())
    _LOG.info("Server started at http://%s:%d (WS at /ws)", host, port)
    return runner, app


def main(args=None):
    # Init rclpy and ROS node
    rclpy.init(args=args)
    node = SocketServer()

    # Use MultiThreadedExecutor in a separate thread
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    def spin_executor():
        try:
            executor.spin()
        except Exception:
            _LOG.exception("Executor spin exception")
        finally:
            _LOG.info("Executor spin finished")

    import threading

    t = threading.Thread(target=spin_executor, daemon=True)
    t.start()
    _LOG.info("rclpy executor thread started")

    # Start asyncio loop in main thread
    loop = asyncio.get_event_loop()

    # Graceful cancel/shutdown handling
    stop_event = asyncio.Event()

    def _shutdown(signame):
        _LOG.info("Received signal %s, shutting down...", signame)
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: _shutdown(s.name))

    try:
        runner, app = run(loop, node)
        # Wait until shutdown requested
        loop.run_until_complete(stop_event.wait())
    except Exception:
        _LOG.exception("Server exception")
    finally:
        _LOG.info("Cleaning up aiohttp runner and rclpy")
        # Cleanup aiohttp
        try:
            loop.run_until_complete(runner.cleanup())
        except Exception:
            _LOG.exception("Error cleaning up runner")

        # Shutdown rclpy/executor
        try:
            executor.shutdown()
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            _LOG.exception("Error shutting down rclpy")
        _LOG.info("Shutdown complete")


if __name__ == "__main__":
    main()
