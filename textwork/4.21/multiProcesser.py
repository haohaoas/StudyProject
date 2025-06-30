import  multiprocessing
def coding():
    for i in range(10):
        print("i'm coding")


def talking():
    for i in range(10):
        print("i'm talking")

def musing():
    for i in range(10):
        print("i'm listening")

if __name__=="__main__":
    multiprocess=multiprocessing.Process(target=coding)
    multiprocess1=multiprocessing.Process(target=talking)
    multiprocess2=multiprocessing.Process(target=musing)
    multiprocess2.start()
    print(multiprocess2.pid)
    multiprocess1.start()
    print(multiprocess1.pid)
    multiprocess.start()
    print(multiprocess.pid)
