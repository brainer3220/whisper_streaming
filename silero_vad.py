import torch
import numpy as np
import soundfile as sf
import librosa
from transformers import Wav2Vec2Processor, HubertModel

# Silero VADIterator 클래스 정의
class VADIterator:
    def __init__(self,
                 model,
                 threshold: float = 0.5,
                 sampling_rate: int = 16000,
                 min_silence_duration_ms: int = 100,
                 speech_pad_ms: int = 30
                 ):

        """
        스트림 모방을 위한 클래스

        Parameters
        ----------
        model: 사전 로드된 .jit silero VAD 모델

        threshold: float (기본값 - 0.5)
            음성 임계값. Silero VAD는 각 오디오 청크에 대한 음성 확률을 출력하며, 이 값 이상인 경우 음성으로 간주됩니다.
            각 데이터셋에 대해 이 매개변수를 조정하는 것이 좋지만, 대부분의 데이터셋에서는 "lazy" 0.5가 꽤 좋습니다.

        sampling_rate: int (기본값 - 16000)
            현재 silero VAD 모델은 8000 및 16000 샘플링 속도를 지원합니다.

        min_silence_duration_ms: int (기본값 - 100 밀리초)
            각 음성 청크의 끝에서 min_silence_duration_ms 동안 대기한 후 분리합니다.

        speech_pad_ms: int (기본값 - 30 밀리초)
            최종 음성 청크는 양쪽에 speech_pad_ms만큼 패딩됩니다.
        """

        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate

        if sampling_rate not in [16000]:
            raise ValueError('VADIterator는 [16000] 이외의 샘플링 속도를 지원하지 않습니다.')

        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        self.reset_states()

    def reset_states(self):

        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    def __call__(self, x, return_seconds=False):
        """
        x: torch.Tensor
            오디오 청크

        return_seconds: bool (기본값 - False)
            타임스탬프를 초 단위로 반환할지 여부
        """

        if not torch.is_tensor(x):
            try:
                x = torch.Tensor(x)
            except:
                raise TypeError("오디오를 텐서로 변환할 수 없습니다. 수동으로 변환하세요.")

        window_size_samples = len(x[0]) if x.dim() == 2 else len(x)
        self.current_sample += window_size_samples

        speech_prob = self.model(x, self.sampling_rate).item()

        if (speech_prob >= self.threshold) and self.temp_end:
            self.temp_end = 0

        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            speech_start = self.current_sample - self.speech_pad_samples
            return {'start': int(speech_start) if not return_seconds else round(speech_start / self.sampling_rate, 1)}

        if (speech_prob < self.threshold - 0.15) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            else:
                speech_end = self.temp_end + self.speech_pad_samples
                self.temp_end = 0
                self.triggered = False
                return {'end': int(speech_end) if not return_seconds else round(speech_end / self.sampling_rate, 1)}

        return None

# FixedVADIterator 클래스 정의 (Silero VAD의 512 샘플 제한 우회)
class FixedVADIterator(VADIterator):

    def reset_states(self):
        super().reset_states()
        self.buffer = np.array([], dtype=np.float32)

    def __call__(self, x, return_seconds=False):
        self.buffer = np.append(self.buffer, x)
        if len(self.buffer) >= 512:
            ret = super().__call__(self.buffer, return_seconds=return_seconds)
            self.buffer = np.array([], dtype=np.float32)
            return ret
        return None

def main():
    # Silero VAD 모델 로드
    print("Silero VAD 모델을 로드하는 중...")
    vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False
    )
    vad_iterator = FixedVADIterator(vad_model)

    # 오디오 파일 로드 (예: 'audio.wav')
    audio_path = 'audio.wav'  # 분석할 오디오 파일 경로
    print(f"오디오 파일을 로드하는 중: {audio_path}")
    audio, sr = sf.read(audio_path)

    # 샘플링 속도가 16000이 아닌 경우 리샘플링
    target_sr = 16000
    if sr != target_sr:
        print(f"샘플링 속도를 {sr}에서 {target_sr}으로 변환하는 중...")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # VAD를 통해 음성 구간 감지
    print("음성 구간을 감지하는 중...")
    chunk_size = 512  # 최소 512 샘플 필요
    speech_segments = []
    current_sample = 0
    speech_start = None

    while current_sample < len(audio):
        chunk = audio[current_sample:current_sample + chunk_size]
        result = vad_iterator(chunk)

        if result is not None:
            if 'start' in result:
                speech_start = result['start']
                print(f"음성 시작: {speech_start} 샘플 ({speech_start / sr} 초)")
            elif 'end' in result and speech_start is not None:
                speech_end = result['end']
                speech_segments.append((speech_start, speech_end))
                print(f"음성 종료: {speech_end} 샘플 ({speech_end / sr} 초)")
                speech_start = None

        current_sample += chunk_size

    # 마지막 남은 음성 구간 처리
    if vad_iterator.triggered and speech_start is not None:
        speech_end = vad_iterator.current_sample
        speech_segments.append((speech_start, speech_end))
        print(f"음성 종료 (마지막): {speech_end} 샘플 ({speech_end / sr} 초)")

    if not speech_segments:
        print("감지된 음성 구간이 없습니다.")
        return

    print(f"총 감지된 음성 구간 수: {len(speech_segments)}")

    # Hubert 모델 로드
    print("Hubert 모델을 로드하는 중...")
    processor = Wav2Vec2Processor.from_pretrained("team-lucid/hubert-base-korean")
    hubert_model = HubertModel.from_pretrained("team-lucid/hubert-base-korean")
    hubert_model.eval()

    # 감지된 음성 구간을 Hubert 모델에 입력하여 특징 벡터 추출
    for idx, (start, end) in enumerate(speech_segments):
        print(f"\n음성 구간 {idx + 1}: {start} 샘플 ({start / sr} 초) ~ {end} 샘플 ({end / sr} 초)")
        speech_chunk = audio[int(start):int(end)]

        # 모델 입력 형식으로 변환
        input_values = processor(speech_chunk, sampling_rate=sr, return_tensors="pt").input_values

        # 모델 추론
        with torch.no_grad():
            outputs = hubert_model(input_values)

        # 출력된 특징 벡터 활용
        hidden_states = outputs.last_hidden_state
        print(f"특징 벡터의 형태: {hidden_states.shape}")  # 예: [1, 49, 768]
        # 여기서 추가적인 후처리 또는 다운스트림 작업을 수행할 수 있습니다.

if __name__ == "__main__":
    main()
