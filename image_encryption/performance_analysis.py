import time

def start_timer():
    return time.perf_counter()


def stop_timer(start_time):
    return time.perf_counter() - start_time


def print_execution_time(encryption_time, decryption_time):
    print("\n========== Performance Analysis ==========")
    print(f"Encryption Time : {encryption_time:.6f} seconds")
    print(f"Decryption Time : {decryption_time:.6f} seconds")
    print(f"Total Time      : {encryption_time + decryption_time:.6f} seconds")