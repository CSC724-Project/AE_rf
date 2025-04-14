import csv

# Parameter pools for simulation
file_sizes = [100, 256, 500, 1024, 2048, 4096, 8192, 10240]     # in MB
block_sizes = [4, 8, 64, 256, 512, 1024, 2048]                   # in KB
jobs_list = [1, 2, 4, 8, 16]
io_depth_list = [4, 8, 16, 32, 64, 128]
patterns = ["randread", "randwrite", "randrw"]
# Candidate optimal chunk sizes, expressed in MB.
optimal_chunk_sizes = [1, 2, 4, 8, 16, 32]

# Open a CSV file to write 100 simulated test cases
with open("simulated_fio_tests.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Write header row (added the optimal_chunk_size field at the end)
    writer.writerow([
        "Test_ID", "File_Size_MB", "Block_Size_KB", "Jobs", "IO_Pattern", "IO_Depth",
        "Direct_IO",
        "IOPS_Read", "Throughput_Read_MiBps", "Avg_Latency_Read_usec",
        "IOPS_Write", "Throughput_Write_MiBps", "Avg_Latency_Write_usec",
        "CPU_User_Percent", "CPU_Sys_Percent",
        "Optimal_Chunk_Size_MB"
    ])
    
    # Generate 100 simulated test cases
    for i in range(1, 101):
        # Cycle through our parameter pools using modulo arithmetic
        fs = file_sizes[(i - 1) % len(file_sizes)]
        bs = block_sizes[(i - 1) % len(block_sizes)]
        jobs = jobs_list[(i - 1) % len(jobs_list)]
        depth = io_depth_list[(i - 1) % len(io_depth_list)]
        pattern = patterns[(i - 1) % len(patterns)]
        direct = "Yes" if i % 2 == 0 else "No"
        # Determine optimal_chunk_size using our predefined candidate list
        optimal_chunk_size = optimal_chunk_sizes[(i - 1) % len(optimal_chunk_sizes)]
        
        # Compute a base IOPS value (this formula is arbitrary and for simulation only)
        base_iops = jobs * depth * (512.0 / bs) + ((i % 10) * 5)
        
        # Calculate IOPS for read and write depending on I/O pattern.
        if pattern == "randread":
            iops_read = int(round(base_iops * (1.2 if direct == "Yes" else 1.0)))
            iops_write = 0
        elif pattern == "randwrite":
            iops_read = 0
            iops_write = int(round(base_iops * (1.2 if direct == "Yes" else 1.0)))
        else:  # "randrw"
            # Split into 60% read and 40% write (and adjust if Direct_IO is enabled)
            factor = 1.2 if direct == "Yes" else 1.0
            iops_read = int(round(base_iops * 0.6 * factor))
            iops_write = int(round(base_iops * 0.4 * factor))
        
        # Compute throughput (MiB/s): each I/O transfers "bs" KB so throughput = (IOPS * bs) / 1024
        thr_read = round(iops_read * bs / 1024, 2)
        thr_write = round(iops_write * bs / 1024, 2)
        
        # Calculate average latency in microseconds (simulated formula)
        lat_read = round(10 + (bs / 128) + (depth / 8) + (0 if direct == "Yes" else 5) + (i % 7), 1) if iops_read > 0 else 0
        lat_write = round(12 + (bs / 128) + (depth / 10) + (0 if direct == "Yes" else 7) + (i % 5), 1) if iops_write > 0 else 0
        
        # Simulate simple CPU usage percentages
        cpu_user = round(jobs * 0.5 + depth / 16, 1)
        cpu_sys = round(jobs * 1.0 + depth / 8, 1)
        
        # Write the simulated test case row to the CSV file, including optimal_chunk_size
        writer.writerow([
            i, fs, bs, jobs, pattern, depth, direct,
            iops_read, thr_read, lat_read,
            iops_write, thr_write, lat_write,
            cpu_user, cpu_sys,
            optimal_chunk_size
        ])

print("CSV file 'simulated_fio_tests.csv' generated successfully!")