#python가상화 설정
python -m venv env

#가상화 활성화
source env/bin/activate

#사전에 필요한 라이브러리를 설치합니다.
pip install -r requirements.txt

#로컬에서 실행할 경우 (가상화 활성화 한뒤)
streamlit run app.py

#pm2에서 실행할 경우
pm2 start app.py --interpreter streamlit --interpreter-args run