#!/usr/bin/env python3
from whisper_online import *

import sys
import argparse
import os
import logging
import numpy as np
import threading  # Import threading module

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

# server options
parser.add_argument("--host", type=str, default='localhost')
parser.add_argument("--port", type=int, default=43007)
parser.add_argument("--warmup-file", type=str, dest="warmup_file", 
        help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast. It can be e.g. https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav .")

# options from whisper_online
add_shared_args(parser)
args = parser.parse_args()

set_logging(args, logger, other="")

# Constants
SAMPLING_RATE = 16000

# Function to handle each client connection
def handle_client(conn, addr):
    logger.info(f'Connected to client on {addr}')

    # Initialize ASR and online instances per connection
    asr, online = asr_factory(args)

    # Warm up the ASR if a warmup file is provided
    if args.warmup_file:
        if os.path.isfile(args.warmup_file):
            a = load_audio_chunk(args.warmup_file, 0, 1)
            asr.transcribe(a)
            logger.info(f"Whisper is warmed up for client {addr}")
        else:
            logger.critical(f"The warm up file is not available for client {addr}. Terminating connection.")
            conn.close()
            return
    else:
        logger.warning("Whisper is not warmed up. The first chunk processing may take longer.")

    connection = Connection(conn)
    proc = ServerProcessor(connection, online, args.min_chunk_size)

    try:
        proc.process()
    except Exception as e:
        logger.error(f"Error processing client {addr}: {e}")
    finally:
        conn.close()
        logger.info(f'Connection to client {addr} closed')

######### Server objects

import line_packet
import socket
import io
import soundfile
import librosa

class Connection:
    '''It wraps conn object'''
    PACKET_SIZE = 32000*5*60  # 5 minutes

    def __init__(self, conn):
        self.conn = conn
        self.last_line = ""

        self.conn.setblocking(True)

    def send(self, line):
        '''It doesn't send the same line twice to prevent duplicates'''
        if line == self.last_line:
            return
        line_packet.send_one_line(self.conn, line)
        self.last_line = line

    def receive_lines(self):
        in_line = line_packet.receive_lines(self.conn)
        return in_line

    def non_blocking_receive_audio(self):
        try:
            r = self.conn.recv(self.PACKET_SIZE)
            return r
        except ConnectionResetError:
            return None

# Wraps socket and ASR object, and serves one client connection.
# Next client should be served by a new instance of this object
class ServerProcessor:

    def __init__(self, c, online_asr_proc, min_chunk):
        self.connection = c
        self.online_asr_proc = online_asr_proc
        self.min_chunk = min_chunk

        self.last_end = None

        self.is_first = True

    def receive_audio_chunk(self):
        # Receive all audio that is available by this time
        # Blocks operation if less than self.min_chunk seconds is available
        # Unblocks if connection is closed or a chunk is available
        out = []
        minlimit = self.min_chunk * SAMPLING_RATE
        while sum(len(x) for x in out) < minlimit:
            raw_bytes = self.connection.non_blocking_receive_audio()
            if not raw_bytes:
                break
            sf = soundfile.SoundFile(io.BytesIO(raw_bytes), channels=1, endian="LITTLE", samplerate=SAMPLING_RATE, subtype="PCM_16", format="RAW")
            audio, _ = librosa.load(sf, sr=SAMPLING_RATE, dtype=np.float32)
            out.append(audio)
        if not out:
            return None
        conc = np.concatenate(out)
        if self.is_first and len(conc) < minlimit:
            return None
        self.is_first = False
        return np.concatenate(out)

    def format_output_transcript(self, o):
        # Output format in stdout is like:
        # 0 1720 Takhle to je
        # - The first two numbers are:
        #    - Begin and end timestamp of the text segment, as estimated by Whisper model.
        #      The timestamps are not accurate, but they're useful anyway.
        # - The next words: segment transcript

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

    def send_result(self, o):
        msg = self.format_output_transcript(o)
        if msg is not None:
            self.connection.send(msg)

    def process(self):
        # Handle one client connection
        self.online_asr_proc.init()
        while True:
            a = self.receive_audio_chunk()
            if a is None:
                break
            self.online_asr_proc.insert_audio_chunk(a)
            o = self.online_asr_proc.process_iter()
            try:
                self.send_result(o)
            except BrokenPipeError:
                logger.info("Broken pipe -- connection closed?")
                break

        # Optionally, finish processing
        # o = self.online_asr_proc.finish()
        # self.send_result(o)

# Server loop with threading
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Reuse address
    s.bind((args.host, args.port))
    s.listen(10)  # Increased backlog to allow more pending connections
    logger.info(f'Listening on {args.host}:{args.port}')

    try:
        while True:
            conn, addr = s.accept()
            client_thread = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            client_thread.start()
    except KeyboardInterrupt:
        logger.info("Server is shutting down.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

logger.info('Server terminated.')
