import socket

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_socket.bind(('127.0.0.1', 9999))
server_socket.listen(5)
print('服务端开始运行了')
accept_socket, client_addr = server_socket.accept()
while True:

    recv_data_bytes = accept_socket.recv(1024)
    recv_data_str = recv_data_bytes.decode('utf8')
    print(f'服务器端收到回执信息:{recv_data_str}')
    if recv_data_str == 'exit':
        break
    accept_socket.close()
# server_socket.close()
# count=0
# while True:
#     count+=1
#     accept_socket, client_addr = server_socket.accept()
#     with open(f'test_{count}.txt', 'wb') as f:
#        while True:
#         data=accept_socket.recv(1024)
#         if not data:
#             break
#         f.write(data)
#     accept_socket.close()
#server_socket.close()
