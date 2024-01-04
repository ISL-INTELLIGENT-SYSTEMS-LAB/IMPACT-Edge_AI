# Import necessary modules
import socket  # For network connections
import pandas as pd  # For data manipulation
import os  # For operating system related tasks
import threading  # For multithreading
from io import StringIO  # For string IO operations

# Define the directory where the CSV files will be saved
DIR = '/home/santino/Desktop/test'  # change to correct directory 
# Define the server address and port
SERVER_ADDRESS = ('192.168.0.21', 16666)  # change to correct server IPv4 address

# Function to process the received data
def process_data(data_bytes, client_address, total_received):
    """
    Processes the received data by decoding it, splitting it into filename and json_str, 
    converting the JSON string to a pandas DataFrame and saving it to a CSV file.
    
    Args:
        data_bytes (bytes): The raw data received from the client.
    """
    # Decode the data and split it into filename and JSON string
    filename, json_str = data_bytes.decode().split('|||', 1)
    # Convert the JSON string to a DataFrame
    df = pd.read_json(StringIO(json_str))
    # If the directory does not exist, create it
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    # Save the DataFrame to a CSV file
    df.to_csv(os.path.join(DIR, f'{filename}.csv'), index=False)
    # Print a message indicating the total amount of data received and the filename
    print(f"\tReceived: {total_received} bytes, {filename}")
    # Print a message indicating the client has disconnected
    print(f"Client {client_address} disconnected.")

# Function to handle a client connection
def handle_client(client_socket, client_address):
    """
    Handles a client connection by receiving data from the client and processing it.
    
    Args:
        client_socket (socket.socket): The socket object associated with the client.
        client_address (tuple): The address of the client.
    """
        # Initialize the data bytes and the total amount of data received
    data_bytes = b''
    total_received = 0
    # Loop until no more data is received from the client
    while True:
        # Receive data from the client
        packet = client_socket.recv(1024)
        # If no data was received, break the loop
        if not packet: 
            break
        # Add the received data to the data bytes
        data_bytes += packet
        # Add the length of the received data to the total
        total_received += len(packet)
    # Process the received data
    process_data(data_bytes, client_address, total_received)
    # Close the client socket
    client_socket.close()

# Function to start the server
def start_server():
    """
    Starts the server, listens for client connections, and starts a new thread for each client that connects.
    """
    # Create a server socket
    server_socket = socket.socket()
    # Bind the server socket to the server address
    server_socket.bind(SERVER_ADDRESS) 
    # Start listening for client connections
    server_socket.listen(5)
    # Print a message indicating the server is listening
    print(f'*** Server {SERVER_ADDRESS} is listening... ***\n')

    # Loop indefinitely
    while True:
        # Accept a client connection
        client_socket, client_address = server_socket.accept()
        # Print a message indicating a client has connected
        print(f"Client {client_address} connected.")
        # Send a message to the client indicating it has connected to the server
        client_socket.send(f'Connected to server {SERVER_ADDRESS}...'.encode('ascii'))
        # Start a new thread to handle the client connection
        threading.Thread(target=handle_client, args=(client_socket, client_address)).start()




if __name__ == "__main__":
    start_server()
