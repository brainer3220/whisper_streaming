#!/usr/bin/env python3
import sys
import argparse
import os
import logging
from logging.handlers import RotatingFileHandler
import structlog
import asyncio
import soundfile
import io
import librosa
from whisper_online import *
import datetime
import uuid
import numpy as np  # Ensure numpy is imported

# Function to configure structlog
def configure_structlog(args):
    """
    Configures structlog with appropriate processors and handlers.
    """
    # Determine logging level based on mode
    if args.mode == "production":
        log_level = logging.INFO
        log_file = "app.log"
    else:
        log_level = logging.DEBUG
        log_file = "app_debug.log"
    
    # Configure standard logging
    logging.basicConfig(
        level=log_level,
        format="%(message)s",  # structlog will handle formatting
        handlers=[
            RotatingFileHandler(
                log_file,
                maxBytes=5 * 1024 * 1024,  # 5 MB
                backupCount=5,
            ),
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,  # Filter logs based on level
            structlog.stdlib.add_logger_name,  # Add logger name
            structlog.stdlib.add_log_level,    # Add log level
            structlog.processors.TimeStamper(fmt="iso"),  # Add timestamp
            structlog.processors.JSONRenderer(),  # Render logs as JSON
            # If you prefer human-readable logs, use ConsoleRenderer
            # structlog.processors.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

# Helper functions for encoding and decoding lines
def encode_line(line: str) -> bytes:
    """Encodes a string into bytes with a newline delimiter."""
    return (line + '\n').encode('utf-8')

def decode_line(data: bytes) -> str:
    """Decodes bytes into a string, stripping the newline delimiter."""
    return data.decode('utf-8').rstrip('\n')

# Argument parsing
parser = argparse.ArgumentParser()

# Example arguments
parser.add_argument("--mode", type=str, default="development", choices=["production", "development"], help="Run mode.")
parser.add_argument("--host", type=str, default='localhost', help="Host to bind the server.")
parser.add_argument("--port", type=int, default=43007, help="Port to bind the server.")
parser.add_argument("--warmup-file", type=str, dest="warmup_file",
                    help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast.")
# Add other arguments as needed
add_shared_args(parser)  # Assuming this function adds more arguments

args = parser.parse_args()

# Configure structlog
configure_structlog(args)

# Obtain a structlog logger
logger = structlog.get_logger(__name__)

# Constants
SAMPLING_RATE = 16000

# Warm up the ASR if a warmup file is provided
msg = "Whisper is not warmed up. The first chunk processing may take longer."
if args.warmup_file:
    if os.path.isfile(args.warmup_file):
        a = load_audio_chunk(args.warmup_file, 0, 1)
        # Initialize a global ASR model for warmup
        global_asr, _ = asr_factory(args)
        global_asr.transcribe(a)
        logger.info("Whisper is warmed up.")
    else:
        logger.critical("The warm up file is not available. " + msg)
        sys.exit(1)
else:
    logger.warning(msg)

######### Server objects

class Connection:
    '''It wraps reader and writer objects for asyncio'''
    PACKET_SIZE = 32000*5*60  # 5 minutes

    def __init__(self, reader, writer):
        self.reader = reader
        self.writer = writer
        self.last_line = ""

    async def send(self, line):
        '''It doesn't send the same line twice to prevent duplicates'''
        if line == self.last_line:
            return
        data = encode_line(line)
        self.writer.write(data)
        await self.writer.drain()
        self.last_line = line

    async def receive_audio(self, min_chunk_size):
        '''Asynchronously receive audio data until min_chunk_size is met'''
        out = []
        minlimit = min_chunk_size * SAMPLING_RATE
        total_received = 0
        while total_received < minlimit:
            data = await self.reader.read(self.PACKET_SIZE)
            if not data:
                break
            total_received += len(data)
            out.append(data)
        if not out:
            return None
        return b''.join(out)

class ServerProcessor:

    def __init__(self, connection, args, addr):
        self.connection = connection
        self.min_chunk = args.min_chunk_size
        self.last_end = None
        self.is_first = True

        # Initialize ASR and online instances per connection
        self.asr, self.online_asr_proc = asr_factory(args)

        # Generate a unique filename using client address and timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = uuid.uuid4().hex  # Ensures uniqueness even if multiple connections from same addr
        client_ip, client_port = addr
        
        AUDIO_DIR = "received_audios"
        os.makedirs(AUDIO_DIR, exist_ok=True)
        self.audio_filename = os.path.join(
            AUDIO_DIR,
            f"audio_{client_ip}_{client_port}_{timestamp}_{unique_id}.wav"
        )

        try:
            self.audio_file = soundfile.SoundFile(
                self.audio_filename,
                mode='w',
                samplerate=SAMPLING_RATE,
                channels=1,
                subtype='PCM_16',
                format='WAV'
            )
            logger.info("Audio file created", filename=self.audio_filename, client=addr)
        except Exception as e:
            logger.error("Failed to create audio file", error=str(e))
            raise

    async def receive_audio_chunk(self):
        raw_bytes = await self.connection.receive_audio(self.min_chunk)
        if not raw_bytes:
            return None
        # Convert raw bytes to audio data
        try:
            sf_obj = soundfile.SoundFile(io.BytesIO(raw_bytes), channels=1, endian="LITTLE",
                                         samplerate=SAMPLING_RATE, subtype="PCM_16", format="RAW")
            audio, _ = librosa.load(sf_obj, sr=SAMPLING_RATE, dtype=np.float32)
        except Exception as e:
            logger.error("Failed to process audio chunk", error=str(e))
            return None

        # Write the audio chunk to the file
        try:
            self.audio_file.write(audio)
        except Exception as e:
            logger.error("Failed to write audio chunk to file", error=str(e))

        if self.is_first and len(audio) < self.min_chunk * SAMPLING_RATE:
            return None
        self.is_first = False
        return audio

    def format_output_transcript(self, o):
        # Output format in stdout is like:
        # 0 1720 Takhle to je
        if o[0] is not None:
            beg, end = o[0]*1000, o[1]*1000
            if self.last_end is not None:
                beg = max(beg, self.last_end)

            self.last_end = end
            # Structured logging as a dictionary
            log_message = {
                "beg": f"{beg:.0f}",
                "end": f"{end:.0f}",
                "transcript": o[2]
            }
            # Log as structured data
            logger.info("Transcription", **log_message)
            # Return a formatted string for sending to the client
            return f"{beg:.0f} {end:.0f} {o[2]}"
        else:
            logger.debug("No text in this segment")
            return None

    async def send_result(self, o):
        msg = self.format_output_transcript(o)
        if msg is not None:
            await self.connection.send(msg)

    async def process(self):
        # Handle one client connection
        self.online_asr_proc.init()
        try:
            while True:
                a = await self.receive_audio_chunk()
                if a is None:
                    break
                self.online_asr_proc.insert_audio_chunk(a)
                o = self.online_asr_proc.process_iter()
                try:
                    await self.send_result(o)
                except ConnectionResetError:
                    logger.info("Connection reset -- connection closed?")
                    break
        except Exception as e:
            logger.error("Error during processing", error=str(e))
        finally:
            # Optionally, finish processing
            # o = self.online_asr_proc.finish()
            # await self.send_result(o)

            # Close the audio file
            try:
                self.audio_file.close()
                logger.info("Audio file closed", filename=self.audio_filename)
            except Exception as e:
                logger.error("Failed to close audio file", error=str(e))

async def handle_client(reader, writer):
    addr = writer.get_extra_info('peername')
    logger.info("New connection", client=addr)

    connection = Connection(reader, writer)
    processor = ServerProcessor(connection, args, addr)

    try:
        await processor.process()
    except Exception as e:
        logger.error("Error processing client", client=addr, error=str(e))
    finally:
        writer.close()
        await writer.wait_closed()
        logger.info("Connection closed", client=addr)

async def main():
    server = await asyncio.start_server(
        handle_client, args.host, args.port
    )
    addr = server.sockets[0].getsockname()
    logger.info("Server started", address=addr)

    async with server:
        try:
            await server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Server is shutting down.")
        except Exception as e:
            logger.error("Server encountered an error", error=str(e))

    logger.info("Server terminated.")

if __name__ == '__main__':
    asyncio.run(main())
