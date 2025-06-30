with open('python.txt', 'w') as w:
    for i in range(10,20):
        w.writelines(str(i) + '\n')
def read_data():
    with open('python.txt', 'r') as f:

        for i in range(1, 11):
            print(f.readline())
if __name__ == '__main__':
    read_data()