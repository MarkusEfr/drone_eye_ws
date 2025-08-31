#!/usr/bin/env python3
import asyncio
import threading
import uuid
from pathlib import Path
import cv2
import logging
from aiohttp import web
from aiohttp.multipart import BodyPartReader
from rclpy.qos import QoSProfile, ReliabilityPolicy

import rclpy
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

_LOG = logging.getLogger("server_node")
logging.basicConfig(level=logging.INFO)

# Session state
SESSIONS = {}  # session_id -> dict
OUTPUT_DIR = Path("/tmp/drone_eye_sessions")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


class ServerNode(Node):
    def __init__(self):
        super().__init__("server_node")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.bridge = CvBridge()

        # Publisher for raw frames (to detector)
        self.pub_raw = self.create_publisher(Image, "/camera/image_raw", qos)

        # Subscriber for visualization (annotated output)
        self.create_subscription(
            Image, "/drone_eye/visualization", self.visual_callback, qos
        )

    async def publish_video(self, session_id: str, path: Path):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            _LOG.error(f"Cannot open video {path}")
            return

        sess = SESSIONS[session_id]
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1.0:
            fps = 30.0
        sess["fps"] = float(fps)
        sess["total_frames"] = total

        delay = 1.0 / fps  # <-- frame spacing in seconds
        idx = 0

        while cap.isOpened() and sess["active"]:
            ret, frame = cap.read()
            if not ret:
                break

            msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            msg.header.frame_id = f"{session_id}#{idx}"
            self.pub_raw.publish(msg)
            idx += 1

            # sleep to maintain original fps timing
            await asyncio.sleep(delay)

        cap.release()
        await finalize_session(session_id)

    def visual_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Extract session_id from header.frame_id: "<session>#<idx>"
        fid = msg.header.frame_id or ""
        session_id = fid.split("#", 1)[0]
        if not session_id:
            return

        sess = SESSIONS.get(session_id)
        if not sess or not sess["active"]:
            return

        # Lazy-init writer exactly once, with frame size and session fps
        if sess["writer"] is None:
            h, w, _ = frame.shape
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            fps = sess.get("fps", 30.0)
            # Guard creation under lock too
            with sess["lock"]:
                if sess["writer"] is None:  # re-check inside lock
                    sess["writer"] = cv2.VideoWriter(
                        str(sess["video_path"]), fourcc, fps, (w, h)
                    )
                    sess["processed_frames"] = 0  # reset processed counter at creation

        # Write frame under lock
        with sess["lock"]:
            if sess["writer"] is not None and sess["active"]:
                sess["writer"].write(frame)
                sess["processed_frames"] += 1


# -------------------------
# HTTP Handlers
# -------------------------
async def handle_upload(request: web.Request):
    reader = await request.multipart()
    field = await reader.next()
    assert isinstance(field, BodyPartReader)

    filename = field.filename or "upload.mp4"
    session_id = str(uuid.uuid4())
    sess_dir = OUTPUT_DIR / session_id
    sess_dir.mkdir(parents=True, exist_ok=True)

    raw_path = sess_dir / filename

    with raw_path.open("wb") as f:
        while True:
            chunk = await field.read_chunk()
            if not chunk:
                break
            f.write(chunk)

    # Init session
    SESSIONS[session_id] = {
        "id": session_id,
        "active": True,
        "raw_path": raw_path,
        "video_path": sess_dir / "out.mp4",
        "writer": None,
        "lock": threading.Lock(),
        "processed_frames": 0,
        "total_frames": 0,
        "fps": 20.0,
    }

    # Start background publishing task
    node: ServerNode = request.app["ros_node"]
    asyncio.create_task(node.publish_video(session_id, raw_path))

    return web.json_response({"session": session_id})


async def handle_status(request: web.Request):
    session_id = request.match_info["session"]
    sess = SESSIONS.get(session_id)
    if not sess:
        return web.json_response({"error": "no such session"}, status=404)

    progress = 0
    total = sess.get("total_frames", 0)
    processed = sess.get("processed_frames", 0)
    if total > 0:
        progress = min(int(processed / total * 100), 100)

    return web.json_response(
        {
            "session": session_id,
            "progress": progress,
            "processed_frames": sess["processed_frames"],
            "total_frames": sess["total_frames"],
            "done": not sess["active"],
            "video": f"/download/{session_id}.mp4" if not sess["active"] else None,
        }
    )


async def handle_download(request: web.Request):
    filename = request.match_info["filename"]
    session_id = filename.split(".")[0]
    sess = SESSIONS.get(session_id)
    if not sess or sess["active"]:
        return web.Response(status=404)

    if not sess["video_path"].exists():
        return web.Response(status=404)

    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return web.FileResponse(path=sess["video_path"], headers=headers)


async def finalize_session(session_id: str):
    sess = SESSIONS.get(session_id)
    if not sess:
        return
    # Flip active first so visual callbacks stop writing
    sess["active"] = False
    # Release writer under lock to avoid races with visual_callback
    with sess["lock"]:
        if sess["writer"] is not None:
            sess["writer"].release()
            sess["writer"] = None
    _LOG.info(f"Session {session_id} finalized")


# -------------------------
# AIOHTTP Application
# -------------------------
async def init_app(node):
    app = web.Application()
    app["ros_node"] = node
    app.router.add_post("/upload", handle_upload)
    app.router.add_get("/status/{session}", handle_status)
    app.router.add_get("/download/{filename}", handle_download)
    # Serve static index.html if present
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.router.add_static("/", str(static_dir), show_index=True)
    return app


def main():
    rclpy.init()
    node = ServerNode()

    # ROS spinning in background
    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()

    loop = asyncio.get_event_loop()
    app = loop.run_until_complete(init_app(node))
    web_runner = web.AppRunner(app)
    loop.run_until_complete(web_runner.setup())
    site = web.TCPSite(web_runner, "0.0.0.0", 8000)
    loop.run_until_complete(site.start())
    _LOG.info("Server running at http://0.0.0.0:8000")
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(web_runner.cleanup())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
