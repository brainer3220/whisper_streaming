#!/usr/bin/env python3
import sys
import argparse
import os
import logging
import numpy as np
import asyncio  # Import asyncio for asynchronous I/O
import soundfile
import io
import librosa

# Assuming whisper_online and asr_factory are correctly imported
from whisper_online import *

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

# Server options
parser.add_argument("--host", type=str, default='localhost')
parser.add_argument("--port", type=int, default=43007)
parser.add_argument("--warmup-file", type=str, dest="warmup_file",
                    help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast. It can be e.g. https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav .")

# Options from whisper_online
add_shared_args(parser)
args = parser.parse_args()

set_logging(args, logger, other="")

# Constants
SAMPLING_RATE = 16000

# Helper functions for encoding and decoding lines
def encode_line(line: str) -> bytes:
    """Encodes a string into bytes with a newline delimiter."""
    return (line + '\n').encode('utf-8')

def decode_line(data: bytes) -> str:
    """Decodes bytes into a string, stripping the newline delimiter."""
    return data.decode('utf-8').rstrip('\n')

# Warm up the ASR because the very first transcribe takes more time than the others.
# Test results in https://github.com/ufal/whisper_streaming/pull/81
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

# Removed line_packet import

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

    def __init__(self, connection, args):
        self.connection = connection
        self.min_chunk = args.min_chunk_size
        self.last_end = None
        self.is_first = True

        # Initialize ASR and online instances per connection
        self.asr, self.online_asr_proc = asr_factory(args)

    async def receive_audio_chunk(self):
        raw_bytes = await self.connection.receive_audio(self.min_chunk)
        if not raw_bytes:
            return None
        # Convert raw bytes to audio data
        sf = soundfile.SoundFile(io.BytesIO(raw_bytes), channels=1, endian="LITTLE",
                                 samplerate=SAMPLING_RATE, subtype="PCM_16", format="RAW")
        audio, _ = librosa.load(sf, sr=SAMPLING_RATE, dtype=np.float32)
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
            print("%1.0f %1.0f %s" % (beg, end, o[2]), flush=True, file=sys.stderr)
            return "%1.0f %1.0f %s" % (beg, end, o[2])
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

        # Optionally, finish processing
        # o = self.online_asr_proc.finish()
        # await self.send_result(o)

async def handle_client(reader, writer):
    addr = writer.get_extra_info('peername')
    logger.info(f'Connected to client on {addr}')

    connection = Connection(reader, writer)
    processor = ServerProcessor(connection, args)

    try:
        await processor.process()
    except Exception as e:
        logger.error(f"Error processing client {addr}: {e}")
    finally:
        writer.close()
        await writer.wait_closed()
        logger.info(f'Connection to client {addr} closed')

async def main():
    server = await asyncio.start_server(
        handle_client, args.host, args.port
    )
    addr = server.sockets[0].getsockname()
    logger.info(f'Listening on {addr}')

    async with server:
        try:
            await server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Server is shutting down.")
        except Exception as e:
            logger.error(f"An error occurred: {e}")

    logger.info('Server terminated.')

if __name__ == '__main__':
    asyncio.run(main())
