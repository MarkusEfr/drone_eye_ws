#!/usr/bin/env python3
import asyncio
import threading
import uuid
from pathlib import Path
import cv2
import numpy as np
import logging
import tempfile
import json
from aiohttp import web
from aiohttp.multipart import MultipartReader, BodyPartReader
from typing import Optional, Union, cast

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from builtin_interfaces.msg import Time as RosTime
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge

from drone_eye_msgs.msg import BoundingBoxes

_LOG = logging.getLogger("drone_eye_server")
logging.basicConfig(level=logging.INFO)

VIDEO_OUTPUT_DIR = Path("/tmp/annotated_videos")
VIDEO_OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------- Helpers ---------------------------


def make_qos():
    # Один профиль на всё, чтобы избежать несовместимости QoS
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=50,
    )


def parse_fid(frame_id_str: str):
    # "sess-xxxx#123" -> ("sess-xxxx", 123)
    try:
        sid, idx = frame_id_str.split("#", 1)
        return sid, int(idx)
    except Exception:
        return "", -1


# --------------------------- ROS Node ---------------------------


class VideoServer(Node):
    def __init__(self):
        super().__init__("drone_eye_server")
        self.bridge = CvBridge()
        qos = make_qos()

        # Publisher: исходные кадры -> детектор
        self.pub_raw = self.create_publisher(Image, "/camera/image_raw", qos)

        # Subscribers: аннотированные кадры + детекции от детектора
        self.create_subscription(
            Image, "/drone_eye/video_results", self.on_annotated, qos
        )
        self.create_subscription(
            BoundingBoxes, "/drone_eye/detections", self.on_dets, qos
        )

        # Состояние по сессиям
        self.sessions = {}  # session_id -> dict with state

        # крутилка времени
        self._clock = self.get_clock()

    # -------------- Состояние сессии --------------

    def _init_session(self, session_id: str, out_fps: int, total_frames: int):
        self.sessions[session_id] = {
            "fps": out_fps,
            "total": total_frames,
            "sent": 0,  # сколько исходных кадров отправлено
            "written": 0,  # сколько аннотированных кадров записано в видео
            "last_frame": None,  # последний аннотированный кадр (np.ndarray)
            "writer": None,  # cv2.VideoWriter
            "size": None,  # (w,h)
            "dets": {},  # frame_index -> list[dict]
            "video_path": str(VIDEO_OUTPUT_DIR / f"{session_id}.mp4"),
            "dets_path": str(VIDEO_OUTPUT_DIR / f"{session_id}.json"),
            "done": False,  # вся исходная подача завершена
            "closed": False,  # файлы закрыты и сохранены
        }

    def _close_session_if_done(self, sid: str):
        st = self.sessions.get(sid)
        if not st or st["closed"]:
            return
        # Закрываем, когда записали все кадры
        if st["written"] >= st["total"]:
            if st["writer"] is not None:
                st["writer"].release()
                st["writer"] = None
            # Заполняем пустые детекции для пропущенных кадров
            total = st["total"]
            dets = st["dets"]
            for i in range(total):
                dets.setdefault(i, [])
            with open(st["dets_path"], "w") as f:
                # Сохраняем в стабильном порядке кадров
                out = [{"frame_id": i, "detections": dets[i]} for i in range(total)]
                json.dump(out, f, indent=2)
            st["closed"] = True
            _LOG.info(
                f"[{sid}] Session closed. Video: {st['video_path']}, Dets: {st['dets_path']}"
            )

    # -------------- Callbacks from detector --------------

    def on_annotated(self, msg: Image):
        # Принимаем аннотированный кадр и кладём его в видео, соблюдая индекс.
        sid, idx = parse_fid(msg.header.frame_id)
        st = self.sessions.get(sid)
        if not st:
            return
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Создаём writer при первом кадре
        if st["writer"] is None:
            h, w = frame.shape[:2]
            st["size"] = (w, h)
            st["writer"] = cv2.VideoWriter(
                st["video_path"],
                cv2.VideoWriter.fourcc(*"mp4v"),
                st["fps"],
                (w, h),
            )
            _LOG.info(
                f"[{sid}] Video writer opened: {st['video_path']} @ {st['fps']} FPS, size={w}x{h}"
            )

        # Дозаполнить пропущенные индексы предыдущим кадром
        while st["written"] < idx:
            if st["last_frame"] is None:
                # Если ещё ничего не было — используем текущий кадр
                st["writer"].write(frame)
                st["written"] += 1
            else:
                st["writer"].write(st["last_frame"])
                st["written"] += 1

        # Записать текущий кадр
        st["writer"].write(frame)
        st["last_frame"] = frame.copy()
        st["written"] = idx + 1

        # Если мы уже отправили все исходные кадры и дописали всё — закрываем
        self._close_session_if_done(sid)

    def on_dets(self, msg: BoundingBoxes):
        sid, idx = parse_fid(msg.header.frame_id)
        st = self.sessions.get(sid)
        if not st:
            return
        dets = []
        for b in msg.boxes:
            dets.append(
                {
                    "id": int(b.id),
                    "label": b.label,
                    "probability": float(b.probability),
                    "xmin": float(b.xmin),
                    "ymin": float(b.ymin),
                    "xmax": float(b.xmax),
                    "ymax": float(b.ymax),
                }
            )
        st["dets"][idx] = dets

        # --- NEW: count processed frames ---
        st["processed_frames"] = st.get("processed_frames", 0) + 1

    # -------------- Публикация исходных кадров --------------

    def _publish_frame(self, session_id: str, frame_index: int, frame: np.ndarray):
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        # Пишем frame_id = "sess#idx"
        fid = f"{session_id}#{frame_index}"
        msg.header = Header()
        msg.header.frame_id = fid
        msg.header.stamp = self._clock.now().to_msg()
        self.pub_raw.publish(msg)

    # -------------- Фоновая обработка одной сессии --------------

    async def _process_video_session(
        self, session_id: str, video_path: str, fps: float, total_frames: int
    ):
        st = self.sessions[session_id]
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            _LOG.error(f"[{session_id}] Failed to open temp video")
            st["done"] = True
            return

        # Публикуем кадры с шагом реального fps (стримингово)
        delay = 1.0 / max(1.0, fps)
        idx = 0
        _LOG.info(f"[{session_id}] Streaming {total_frames} frames @ {fps:.2f} FPS")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self._publish_frame(session_id, idx, frame)
            st["sent"] = idx + 1
            idx += 1
            await asyncio.sleep(delay)  # темп вывода
        cap.release()
        st["done"] = True
        _LOG.info(f"[{session_id}] Source streaming done. Sent={st['sent']}")

        # Если аннотированный видеофайл ещё не дописан — дожмём оставшиеся кадры
        # (будут продублированы последние известные аннотированные кадры)
        # Дождёмся немного приход аннотаций после завершения подачи
        soft_wait_s = min(10.0, total_frames * 0.1)
        end_wait_start = asyncio.get_event_loop().time()
        while (not st["closed"]) and (
            asyncio.get_event_loop().time() - end_wait_start < soft_wait_s
        ):
            # Если детектор всё отдал — прервём ожидание
            if st["written"] >= st["total"]:
                break
            await asyncio.sleep(0.1)

        # Если всё ещё не закрыто — дописываем остаток повтором last_frame
        if not st["closed"]:
            if st["writer"] is not None:
                while st["written"] < st["total"]:
                    if st["last_frame"] is None:
                        break  # совсем не пришёл ни один аннотированный кадр
                    st["writer"].write(st["last_frame"])
                    st["written"] += 1
            self._close_session_if_done(session_id)


# --------------------------- HTTP Handlers ---------------------------


async def upload_handler(request: web.Request):
    node: VideoServer = request.app["ros_node"]

    reader: MultipartReader = await request.multipart()
    raw_field: Union[MultipartReader, BodyPartReader, None] = await reader.next()

    # Берём только BodyPartReader
    file_field: Optional[BodyPartReader] = (
        raw_field if isinstance(raw_field, BodyPartReader) else None
    )
    if not file_field:
        return web.Response(status=400, text="Invalid upload (no file field)")

    # Сохраняем файл во временный путь
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        temp_path = tmp.name
        while True:
            chunk = await file_field.read_chunk()
            if not chunk:
                break
            tmp.write(chunk)

    # Читаем метаданные видео
    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        return web.Response(status=400, text="Invalid video file")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 20.0)
    cap.release()

    # Создаём новую сессию
    session_id = f"sess-{uuid.uuid4().hex[:8]}"
    node._init_session(session_id, int(round(fps)), total_frames)

    # Добавляем прогресс для отслеживания
    node.sessions[session_id].update(
        {
            "done": False,
            "progress": 0,
            "processed_frames": 0,
            "total_frames": total_frames,
        }
    )

    _LOG.info(f"[{session_id}] video has {total_frames} frames @ {fps:.2f} FPS")

    # Запускаем обработку в фоне
    asyncio.create_task(
        node._process_video_session(session_id, temp_path, fps, total_frames)
    )

    # Отдаём ответ сразу
    return web.json_response(
        {
            "session": session_id,
            "status": f"/status/{session_id}",  # для фронта
            "result": f"/result/{session_id}",  # результат по готовности
        },
        status=202,
    )


async def status_handler(request: web.Request):
    node: VideoServer = request.app["ros_node"]
    sid = request.match_info["session"]
    st = node.sessions.get(sid)
    if not st:
        return web.json_response({"error": "unknown session"}, status=404)

    processed = st.get("processed_frames", 0)
    total = st.get("total_frames", st.get("total", 1))
    progress = int((processed / total) * 100) if total else 0

    return web.json_response(
        {
            "session": sid,
            "progress": progress,
            "processed_frames": processed,
            "total_frames": total,
            "done": st.get("closed", False),
            "video": (
                f"/download/{Path(st['video_path']).name}" if st.get("closed") else None
            ),
            "detections": (
                f"/download/{Path(st['dets_path']).name}" if st.get("closed") else None
            ),
        }
    )


async def result_handler(request: web.Request):
    node: VideoServer = request.app["ros_node"]
    sid = request.match_info["session"]
    st = node.sessions.get(sid)
    if not st:
        return web.json_response({"error": "unknown session"}, status=404)
    if not st["closed"]:
        return web.json_response({"status": "processing"}, status=202)
    return web.json_response(
        {
            "session": sid,
            "video": f"/download/{Path(st['video_path']).name}",
            "detections": f"/download/{Path(st['dets_path']).name}",
        }
    )


async def download_handler(request: web.Request):
    filename = request.match_info["filename"]
    file_path = VIDEO_OUTPUT_DIR / filename
    if file_path.exists():
        return web.FileResponse(
            file_path,
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )
    return web.Response(status=404, text="File not found")


async def init_app(node):
    app = web.Application()
    app["ros_node"] = node
    app.router.add_post("/upload", upload_handler)
    app.router.add_get("/status/{session}", status_handler)
    app.router.add_get("/result/{session}", result_handler)
    app.router.add_get("/download/{filename}", download_handler)
    # Статик (если нужен index.html)
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.router.add_static("/", str(static_dir), show_index=True)
    return app


def main():
    rclpy.init()
    node = VideoServer()

    # Спин ROS в фоне
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
