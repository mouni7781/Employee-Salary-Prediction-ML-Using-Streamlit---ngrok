from pyngrok import ngrok
import time
import os
import threading
def run_app():
    os.system(r'streamlit run c:\Users\mouni\OneDrive\Desktop\Resources\Pyton-AI\Project\Application.py --server.port 3000')
thread=threading.Thread(target=run_app)
thread.start()

time.sleep(5)
public_url = ngrok.connect(3000)
print("url to deploy is " ,public_url)
