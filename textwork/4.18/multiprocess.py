import multiprocessing
import time
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor


# def coding():
#     for i in range(10):
#         print("i'm coding")
#         time.sleep(0.1)
# def reading():
#     for i in range(10):
#         print("i'm reading")
#         time.sleep(0.1)
# def music():
#     for i in range(10):
#         print("i'm listening")
#         time.sleep(0.1)
#
# def main():
#     multiprocessing.Process(target=coding).start()
#     multiprocessing.Process(target=reading).start()
#     multiprocessing.Process(target=music).start()

# def coding(num, name):
#     for i in range(num):
#         print(f'{name}正在编写第{i}行代码')
#         time.sleep(0.1)
# def music(count, name):
#     for i in range(count):
#         print(f'{name}正在播放第{i}首歌')
#         time.sleep(0.1)
# def main():
#     multiprocessing.Process(target=coding, args=(10, '小明')).start()
#     multiprocessing.Process(target=music, kwargs={'count': 10, 'name': '小红'}).start()
my_list=[]

def write():
    for i in range(10):
        my_list.append(i)
        print(f'写入数据{i}')
print(my_list)
def read():
        print(f'读取数据{my_list}')
def main():
    p1=multiprocessing.Process(target=write)
    p2=multiprocessing.Process(target=read)
    p1.start()
    time.sleep(0.1)
    p2.start()
    p1.join()
    p2.join()
if __name__ == "__main__":

    main()