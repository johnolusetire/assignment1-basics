import time
import multiprocessing

def cpu_bound_task(n):
    # Example: compute sum of squares up to n
    return sum(i * i for i in range(n))

def run_sequential(tasks, n):
    results = []
    for _ in range(tasks):
        results.append(cpu_bound_task(n))
    return results

def run_multiprocessing(tasks, n):
    print(f"Using {multiprocessing.cpu_count()} CPU cores for multiprocessing.")
    with multiprocessing.Pool() as pool:
        results = pool.map(cpu_bound_task, [n] * tasks)
    return results

if __name__ == "__main__":
    TASKS = 8
    N = 10_000_000

    print("Running sequentially...")
    start = time.time()
    run_sequential(TASKS, N)
    print(f"Sequential time: {time.time() - start:.2f} seconds")

    print("Running with multiprocessing...")
    start = time.time()
    run_multiprocessing(TASKS, N)
    print(f"Multiprocessing time: {time.time() - start:.2f} seconds")