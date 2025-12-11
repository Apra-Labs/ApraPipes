import struct
import sys
import os

def visualize_vad(filename):
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return

    print(f"Reading VAD output from: {filename}")
    
    with open(filename, 'rb') as f:
        data = f.read()

   
    num_samples = len(data) // 4
    print(f"Total Frames: {num_samples}")
    
    valid_bytes_len = num_samples * 4
    if len(data) != valid_bytes_len:
        print(f"Warning: File size {len(data)} is not a multiple of 4. Truncating {len(data) - valid_bytes_len} bytes.")
        data = data[:valid_bytes_len]

   
    values = struct.unpack(f'<{num_samples}i', data)


    zeros = values.count(0)
    ones = values.count(1)
    
    print("-" * 40)
    print(f"Silence Frames (0): {zeros}")
    print(f"Speech Frames  (1): {ones}")
    if num_samples > 0:
        print(f"Speech Activity:    {(ones / num_samples) * 100:.2f}%")
    print("-" * 40)

    chunk_size = 10 
    print(f"\nTimeline (each char = {chunk_size} frames = {chunk_size*10}ms):")
    print("Legend: '_' = Silence, '#' = Speech, '.' = Mixed")
    
    timeline = ""
    for i in range(0, num_samples, chunk_size):
        chunk = values[i:i+chunk_size]
        chunk_sum = sum(chunk)
        
        if chunk_sum == 0:
            timeline += "_"
        elif chunk_sum == len(chunk):
            timeline += "#"
        else:
            timeline += "."
            
        if len(timeline) >= 64:
            print(timeline)
            timeline = ""
            
    if timeline:
        print(timeline)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_vad.py <path_to_raw_file>")
        print("Example: python view_vad.py vad_output.raw")
        
        default_path = "data/vad_output.raw"
        if os.path.exists(default_path):
            print(f"\nNo file specified. Found default: {default_path}")
            visualize_vad(default_path)
    else:
        visualize_vad(sys.argv[1])
