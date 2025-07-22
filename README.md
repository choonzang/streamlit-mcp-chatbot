# 설정

.env는 .env.sample을 참고하셔서, 생성해주세요.

mcp 중 youtube_transcript를 위한 youtube.py 와 같은 디렉토리에도 .env 파일이 필요합니다.

## python가상화 설정
```bash
python -m venv env
```

## 가상화 활성화
```bash
source env/bin/activate
```

## 라이브러리 설치
사전에 필요한 라이브러리를 설치합니다.
```bash
pip install -r requirements.txt
```

## 로컬에서 실행할 경우 (가상화 활성화 한뒤)
```bash
streamlit run app.py
```

## pm2에서 실행할 경우
```bash
pm2 start app.py --interpreter streamlit --interpreter-args run
```