import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import base64
import numpy as np
from aiohttp import web


class VideoProcessor(Node):
    def __init__(self):
        super().__init__("video_processor")
        self.publisher = self.create_publisher(Image, "/camera/image_raw", 10)
        self.bridge = CvBridge()
        self.get_logger().info("WebSocket Video Receiver started.")

    async def handle_ws(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.get_logger().info("Client connected via WebSocket")

        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                # Получаем base64 строку и конвертируем в OpenCV
                try:
                    img_data = base64.b64decode(msg.data.split(",")[1])
                    np_arr = np.frombuffer(img_data, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                    ros_img = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                    self.publisher.publish(ros_img)
                except Exception as e:
                    self.get_logger().error(f"Frame decode error: {e}")
            elif msg.type == web.WSMsgType.ERROR:
                self.get_logger().error(
                    f"WS connection closed with exception {ws.exception()}"
                )

        self.get_logger().info("WebSocket connection closed")
        return ws


def main(args=None):
    rclpy.init(args=args)
    node = VideoProcessor()

    # Запуск aiohttp сервера
    app = web.Application()
    app.router.add_get("/ws", node.handle_ws)

    runner = web.AppRunner(app)
    rclpy.get_default_context().on_shutdown(runner.cleanup)

    async def run_server():
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", 8765)
        await site.start()
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

    import asyncio

    asyncio.run(run_server())


if __name__ == "__main__":
    main()
