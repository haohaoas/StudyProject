import time
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor


def coding():
    for i in range(10):
        print("i'm coding")
        time.sleep(0.1)
def reading():
    for i in range(10):
        print("i'm reading")
        time.sleep(0.1)
def music():
    for i in range(10):
        print("i'm listening")
        time.sleep(0.1)

def main():
    # 创建一个包含5个线程的线程池
    with ThreadPoolExecutor(max_workers=5) as executor:
        # 提交任务到线程池
        future_coding=executor.submit(coding)
        future_reading=executor.submit(reading)
        future_music=executor.submit(music)
        for future in as_completed([future_coding,future_reading,future_music]):
            # 等待任务完成
            future.result()

if __name__ == "__main__":
    main()