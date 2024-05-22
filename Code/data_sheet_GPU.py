import pycuda.driver as cuda
import pycuda.autoinit

def get_gpu_info():
    device_count = cuda.Device.count()
    print(f"Found {device_count} CUDA device(s).")
    
    for device_index in range(device_count):
        device = cuda.Device(device_index)
        print(f"\nDevice {device_index}: {device.name()}")
        print(f"  Compute Capability: {device.compute_capability()}")
        print(f"  Total Memory: {device.total_memory() / 1024**3:.2f} GB")
        
        attributes = device.get_attributes()
        
        # Get specific attributes related to threads and blocks
        max_threads_per_block = attributes[cuda.device_attribute.MAX_THREADS_PER_BLOCK]
        max_block_dim_x = attributes[cuda.device_attribute.MAX_BLOCK_DIM_X]
        max_block_dim_y = attributes[cuda.device_attribute.MAX_BLOCK_DIM_Y]
        max_block_dim_z = attributes[cuda.device_attribute.MAX_BLOCK_DIM_Z]
        max_grid_dim_x = attributes[cuda.device_attribute.MAX_GRID_DIM_X]
        max_grid_dim_y = attributes[cuda.device_attribute.MAX_GRID_DIM_Y]
        max_grid_dim_z = attributes[cuda.device_attribute.MAX_GRID_DIM_Z]
        max_threads_per_sm = attributes[cuda.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR]
        multi_processor_count = attributes[cuda.device_attribute.MULTIPROCESSOR_COUNT]

        print(f"  Max Threads per Block: {max_threads_per_block}")
        print(f"  Max Block Dimensions: ({max_block_dim_x}, {max_block_dim_y}, {max_block_dim_z})")
        print(f"  Max Grid Dimensions: ({max_grid_dim_x}, {max_grid_dim_y}, {max_grid_dim_z})")
        print(f"  Max Threads per SM: {max_threads_per_sm}")
        print(f"  Number of SMs: {multi_processor_count}")

if __name__ == "__main__":
    get_gpu_info()