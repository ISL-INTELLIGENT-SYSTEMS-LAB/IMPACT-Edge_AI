import socket
import pandas as pd
import os
import threading
from io import StringIO


DIR = '/home/santino/Desktop/test' # change to correct directory 
SERVER_ADDRESS = ('192.168.0.21', 16666) # change to correct server IPv4 address



def process_data(data_bytes, client_address, total_received):

    """
    Processes the received data by decoding it, splitting it into filename and json_str, 
    converting the JSON string to a pandas DataFrame and saving it to a CSV file.
    
    Args:
        data_bytes (bytes): The raw data received from the client.
    """

    filename, json_str = data_bytes.decode().split('|||', 1)
    df = pd.read_json(StringIO(json_str))
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    df.to_csv(os.path.join(DIR, f'{filename}.csv'), index=False)
    print(f"\tReceived: {total_received} bytes, {filename}")
    print(f"Client {client_address} disconnected.")


def handle_client(client_socket, client_address):

    """
    Handles a client connection by receiving data from the client and processing it.
    
    Args:
        client_socket (socket.socket): The socket object associated with the client.
        client_address (tuple): The address of the client.
    """

    data_bytes = b''
    total_received = 0
    while True:
        packet = client_socket.recv(1024)
        if not packet: 
            break
        data_bytes += packet
        total_received += len(packet)
    process_data(data_bytes, client_address, total_received)
    client_socket.close()



def start_server():

    """
    Starts the server, listens for client connections, and starts a new thread for each client that connects.
    """

    server_socket = socket.socket()
    server_socket.bind(SERVER_ADDRESS) 
    server_socket.listen(5)
    print(f'*** Server {SERVER_ADDRESS} is listening... ***\n')

    while True:
        client_socket, client_address = server_socket.accept()
        print(f"Client {client_address} connected.")
        client_socket.send(f'Connected to server {SERVER_ADDRESS}...'.encode('ascii'))
        threading.Thread(target=handle_client, args=(client_socket, client_address)).start()



if __name__ == "__main__":
    start_server()

