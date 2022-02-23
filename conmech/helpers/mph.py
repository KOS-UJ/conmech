
from multiprocessing import Process, Queue


def run_processes(function, function_args, num_workers):
    queue = Queue()

    processes = [
        Process(target=lambda *args: queue.put(function(*args)), args=function_args + (num_workers, process_id))
        for process_id in range(num_workers)
    ]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    for _ in processes:
        done = queue.get()
        if not done:
            return False

    return True



def run_process(function): #, args
    queue = Queue()

    #wrapper = lambda *args : queue.put(function())
    wrapper = lambda: queue.put(function())

    process = Process(target=wrapper)
    process.start()
    process.join()
    result = queue.get()

    return result