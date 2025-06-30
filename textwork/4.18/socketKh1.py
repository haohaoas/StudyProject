import socket
socket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
socket.connect(('127.0.0.1',9999))
recv_data_bytes=socket.recv(1024)
recv_data=recv_data_bytes.decode('utf-8')
print(f'服务器端发送的信息:{recv_data}')
socket.send('收到，我是学生1号'.encode('utf-8'))
socket.close()