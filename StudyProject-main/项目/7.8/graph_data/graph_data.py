import json


def print_data_info(data_path):
    i = 0
    with open(data_path, 'r',encoding='utf8') as f:
        lines = f.readlines()
        print(f'数据长度{len(lines)}')
        for line in lines:
            data=json.loads(line)
            print(json.dumps(data,sort_keys=True,indent=4,ensure_ascii=False,separators=(', ', ': ')))
            i+=1
            if i>5:
                break

if __name__ == '__main__':
    data_path = './medical.json'
    print_data_info(data_path)