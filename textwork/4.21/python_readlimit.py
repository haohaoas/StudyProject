def read_data():
    with open('python.txt', 'r') as f:
        while True:
            lines = [f.readline().strip() for _ in range(8)]
            if not any(lines):
                break
            yield lines

if __name__ == '__main__':
    for batch in read_data():
        print(batch)
