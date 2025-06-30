import multiprocess
def task_function():
    for i in range(10):
        print(f"子进程1 PID: {multiprocessing.current_process().pid}")
def task_function2():
    for i in range(10):
        print(f"子进程2 PID: {multiprocessing.current_process().pid}")

# 创建并启动进程
if __name__ == "__main__":
    process = multiprocessing.Process(target=task_function)
    process2 = multiprocessing.Process(target=task_function2)
    process.start()
    process2.start()
    process.join()  # 等待子进程完成
