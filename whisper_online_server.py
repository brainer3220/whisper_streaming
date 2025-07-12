#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Async Whisper streaming server (refactored & de-duplicated).

주요 기능
---------
1) Whisper 모델 싱글톤(ASRPool) 활용으로 메모리 절감
2) structlog JSON 로깅
3) --save-audio 플래그로 선택적 WAV 저장 (빈 파일 자동 삭제)
4) DoS 보호 (세션 당 바이트 한도)
5) conflict 없는 argparse (min-chunk-size는 whisper_online이 관리)
"""
import sys, os, io, uuid, argparse, asyncio, logging, datetime
from logging.handlers import RotatingFileHandler
from typing import Optional

import numpy as np
import structlog
import soundfile as sf

# ────────────────────────────────────────────── Whisper 연동 ──
from whisper_online import asr_factory, add_shared_args, OnlineASRProcessor
# ──────────────────────────────────────────────────────────────

# ────────────────────────────── 상수 정의 ────────────────────
SAMPLING_RATE       = 16_000
PCM16_MAX_ABS       = 2**15         # 32768
PACKET_SIZE         = 32_000 * 5 * 60      # 5분 분량
MAX_SESSION_BYTES   = 32_000 * 5 * 60 * 3  # ≒ 30 MB
AUDIO_DIR           = "received_audios"
# ──────────────────────────────────────────────────────────────

# ╭─────────────────── structlog 설정 함수 ───────────────────╮
def configure_structlog(run_mode: str):
    log_level = logging.INFO if run_mode == "production" else logging.DEBUG
    log_file  = "app.log" if run_mode == "production" else "app_debug.log"

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[
            RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3),
            logging.StreamHandler(sys.stdout)
        ],
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

logger = structlog.get_logger(__name__)

# ╭────────────────────────── Whisper 싱글톤 ───────────────╮
class ASRPool:
    _lock = asyncio.Lock()
    _shared_asr = None

    @classmethod
    async def get_shared_asr(cls, args):
        async with cls._lock:
            if cls._shared_asr is None:
                logger.info("Loading global Whisper model (cold-start)")
                cls._shared_asr, _ = asr_factory(args)
            return cls._shared_asr

# ╭────────────────────────── 헬퍼 ──────────────────────────╮
def bytes_to_audio(raw: bytes) -> np.ndarray:
    """
    16-bit PCM(LITTLE-ENDIAN) → float32 -1~1 정규화.
    """
    i16 = np.frombuffer(raw, dtype="<i2")
    return i16.astype(np.float32) / PCM16_MAX_ABS

def safe_filename() -> str:
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    uid = uuid.uuid4().hex
    return f"audio_{ts}_{uid}.wav"

# ╭──────────────────── Argument Parser ────────────────────╮
parser = argparse.ArgumentParser(
    description="Whisper streaming server (asyncio)"
)
parser.add_argument("--mode", default="development",
                    choices=["production", "development"])
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=43007)
parser.add_argument("--warmup-file")
parser.add_argument("--save-audio", action="store_true",
                    help="Save incoming audio to disk (WAV)")
# ★ min-chunk-size 는 여기서 정의하지 않는다!
add_shared_args(parser)  # whisper_online 공용 인자 포함(min-chunk-size 등)
args = parser.parse_args()

configure_structlog(args.mode)
logger.info("Args parsed", **vars(args))

# ╭──────────────────── 서버 워밍업 ─────────────────────────╮
async def warmup():
    _ = await ASRPool.get_shared_asr(args)
    if args.warmup_file and os.path.isfile(args.warmup_file):
        logger.info("Warm-up transcription start")
        audio = bytes_to_audio(open(args.warmup_file, "rb").read())
        _ , online = asr_factory(args, asr=_)
        online.init()
        online.insert_audio_chunk(audio)
        online.process_iter()
        logger.info("Warm-up done")
    else:
        logger.warning("No warm-up file; first request may be slower")

# ╭───────────────────── Connection 래퍼 ─────────────────────╮
class Connection:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader, self.writer = reader, writer
        self.last_line = ""
        self.total_bytes = 0

    async def send(self, line: str):
        if line != self.last_line:
            self.writer.write((line + "\n").encode())
            await self.writer.drain()
            self.last_line = line

    async def receive_audio(self, min_chunk_sec: float) -> Optional[bytes]:
        needed = int(min_chunk_sec * SAMPLING_RATE * 2)
        chunks, received = [], 0
        while received < needed:
            data = await self.reader.read(PACKET_SIZE)
            if not data:
                break
            received += len(data)
            self.total_bytes += len(data)
            if self.total_bytes > MAX_SESSION_BYTES:
                raise RuntimeError("Session byte quota exceeded")
            chunks.append(data)
        return b"".join(chunks) if chunks else None

# ╭─────────────────── Client Processor ─────────────────────╮
class ServerProcessor:
    def __init__(self, conn: Connection, addr):
        self.conn  = conn
        self.addr  = addr
        self.last_end  = None
        self.is_first  = True
        self.min_chunk = args.min_chunk_size

        # 오디오 저장 초기화
        self.save_audio = args.save_audio
        if self.save_audio:
            os.makedirs(AUDIO_DIR, exist_ok=True)
            self.audio_path = os.path.join(AUDIO_DIR, safe_filename())
            self.audio_file = sf.SoundFile(
                self.audio_path, mode="w",
                samplerate=SAMPLING_RATE, channels=1, subtype="PCM_16")
            logger.info("Audio file opened", client=addr, file=self.audio_path)

        self.online_asr: Optional[OnlineASRProcessor] = None

    async def _lazy_init_asr(self):
        """
        공유 Whisper 모델을 받아 세션 전용 OnlineASRProcessor 생성
        (asr_factory 로 새 모델을 로드하지 않음)
        """
        if self.online_asr is None:
            shared_asr = await ASRPool.get_shared_asr(args)
            # OnlineASRProcessor 시그니처: (whisper_model, args)
            self.online_asr = OnlineASRProcessor(shared_asr, args)
            self.online_asr.init()


    async def _write_audio(self, audio_np: np.ndarray):
        if self.save_audio:
            try:
                self.audio_file.write((audio_np * PCM16_MAX_ABS).astype("<i2"))
            except Exception as e:
                logger.error("Failed to write audio", err=str(e))

    def _format_seg(self, seg):
        if seg[0] is None:
            return None
        beg_ms = max(seg[0]*1000, (self.last_end or 0))
        end_ms = seg[1]*1000
        self.last_end = end_ms
        return f"{beg_ms:.0f} {end_ms:.0f} {seg[2]}"

    async def run(self):
        try:
            await self._lazy_init_asr()
            while True:
                raw = await self.conn.receive_audio(self.min_chunk)
                if not raw:
                    break
                audio_np = bytes_to_audio(raw)
                if self.is_first and len(audio_np) < self.min_chunk * SAMPLING_RATE:
                    continue
                self.is_first = False

                self.online_asr.insert_audio_chunk(audio_np)
                seg = self.online_asr.process_iter()

                msg = self._format_seg(seg)
                if msg:
                    await self.conn.send(msg)
                await self._write_audio(audio_np)

        except RuntimeError as dos:
            logger.warning("DoS triggered", client=self.addr, err=str(dos))
        except asyncio.CancelledError:
            logger.info("Client task cancelled", client=self.addr)
        except Exception as e:
            logger.error("Unhandled error", client=self.addr, err=str(e))
        finally:
            await self._cleanup()

    async def _cleanup(self):
        if self.online_asr:
            try:
                self.online_asr.finish()
            except Exception:
                pass
        if self.save_audio:
            try:
                self.audio_file.close()
                if os.path.getsize(self.audio_path) == 0:
                    os.remove(self.audio_path)
                    logger.info("Removed empty audio", file=self.audio_path)
                else:
                    logger.info("Closed audio file", file=self.audio_path)
            except Exception as e:
                logger.error("Audio close/remove failed", err=str(e))

# ╭──────────────────── TCP 핸들러 ──────────────────────────╮
async def handle_client(reader, writer):
    addr = writer.get_extra_info("peername")
    logger.info("Client connected", client=addr)
    conn = Connection(reader, writer)
    proc = ServerProcessor(conn, addr)
    try:
        await proc.run()
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass
        logger.info("Client disconnected", client=addr)

# ╭────────────────────── 메인 루프 ──────────────────────────╮
async def main():
    await warmup()
    server = await asyncio.start_server(handle_client, args.host, args.port)
    srv_addr = server.sockets[0].getsockname()
    logger.info("Server started", address=srv_addr)

    async with server:
        try:
            await server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Graceful shutdown (KeyboardInterrupt)")
        finally:
            server.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
