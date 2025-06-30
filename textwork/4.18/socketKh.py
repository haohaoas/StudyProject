import socket
socket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
socket.connect(('127.0.0.1', 9999))
while True:
    send_data=input('请输入要发送的信息:')
    socket.send(send_data.encode('utf-8'))
    if send_data=='88':
        break
# with open('test1.txt','rb') as f:
#     while True:
#         data=f.read(1024)
#         if not data:
#             break
#         socket.send(data)
# socket.close()