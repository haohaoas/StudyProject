import threading

g_num = 0


def coding():
    for i in range(10):
        print("i'm coding")


def talking():
    for i in range(10):
        print("i'm talking")

def musing():
    for i in range(10):
        print("i'm listening")


if __name__ == '__main__':
    t1 = threading.Thread(target=coding)
    t2 = threading.Thread(target=talking())
    t3 = threading.Thread(target=musing)

    t2.start()
    t1.run()
    t3.start()