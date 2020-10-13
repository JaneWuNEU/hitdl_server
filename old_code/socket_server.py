import socket
import threading
import datetime
import time
import sys
def conn_port():
    """
    send data to sig01
    :return:
    """
    sig01_ip = "172.16.71.120"
    sig01_port = 10990
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    client.connect((sig01_ip, sig01_port))
    client.send(bytes("server return results",encoding="utf-8"))
def listen_port(ports):
    """
    listen to the data from sig01
    :return:
    """
    ADDR = ("192.168.1.16", ports)
    listen_sock = socket.socket()
    # listen_sock.settimeout(5.0) #设定超时时间后，socket其实内部变成了非阻塞，但有一个超时时间
    listen_sock.bind(ADDR)
    listen_sock.listen(20)
    while True:
        try:
            conn, addr = listen_sock.accept()
            while True:
                a = time.time()
                data = conn.recv(102400)
                b = time.time()
                if len(data)==0:
                    break
                else:
                    print("length",len(data),"period",b-a,"speed",len(data)*8/1024/1024/(b-a),"Mbit/s")
            #print(sys.getsizeof(data),type(data),sys.getsizeof(data)*8/1024/1024.0/(b-a),"Mb/s")
            #print(str(data),conn.getpeername())

        except Exception as e:
            print(e)
#user = threading.Thread(target=listen_port)
#user.start()
user = threading.Thread(target=listen_port,args=[10990])
user.start()

user = threading.Thread(target=listen_port,args=[10991])
user.start()