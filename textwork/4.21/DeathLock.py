import threading
lock = threading.Lock()
def coding():
    lock.acquire()
    for i in range(10):
        print("i'm coding")

def talking():
    lock.acquire()
    for i in range(10):
        print("i'm talking")
    lock.release()

def musing():
    lock.acquire()
    for i in range(10):
        print("i'm listening")
    lock.release()

t1 = threading.Thread(target=coding)
t2 = threading.Thread(target=talking)
t3 = threading.Thread(target=musing)
t2.start()
t1.run()
t3.start()