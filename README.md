# VMR Whisper Streaming 실행

Whisper 스트리밍 서버를 설정하고 실행하려면 다음 단계를 따르세요:

설정을 시작하기 전에 [VMR-Server의 README.md](https://github.com/brainer3220/VMR-Server)를 참고하세요.

1. **VMR-Server 리포지토리 클론**

   먼저, VMR-Server 디렉토리로 이동하여 원하는 리포지토리를 클론합니다:

   ```bash
   cd VMR-Server
   git clone https://github.com/brainer3220?tab=repositories
   ```

2. **`whisper_streaming`**\*\* 디렉토리로 이동\*\*

   서버 실행을 준비하기 위해 `whisper_streaming` 디렉토리로 이동합니다:

   ```bash
   cd whisper_streaming
   ```

3. **Whisper 온라인 서버 실행**

   다음 옵션을 사용하여 Poetry로 Whisper 온라인 서버 스크립트를 실행합니다:

   - 언어를 한국어로 설정 (`--language ko`)
   - 최소 청크 크기를 1로 설정 (`--min-chunk-size 1`)
   - large-v3-turbo 모델 사용 (`--model large-v3-turbo`)
   - 서버를 `0.0.0.0`에서 호스팅 (`--host 0.0.0.0`)
   - 모드를 프로덕션으로 설정 (`--mode production`)

   다음 명령어를 실행합니다:

   ```bash
   poetry run python whisper_online_server.py --language ko --min-chunk-size 1 --model large-v3-turbo --host 0.0.0.0 --mode production
   ```

이렇게 하면 Whisper 스트리밍 서버가 설정되고 네트워크에서 접근 가능해집니다.

