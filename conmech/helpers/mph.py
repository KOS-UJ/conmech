"""
multiprocessing helpers
"""
import sys
from multiprocessing import Lock, Process, Queue
from queue import Empty
from typing import Callable, Tuple


def get_lock():
    return Lock()


def is_supported():
    return "linux" in sys.platform


if not is_supported():
    print("Warning: Multiprocessing implemented only for Linux")


def run_processes(function: Callable, num_workers: int, function_args: Tuple = ()):
    if not is_supported() or num_workers == 1:
        args = function_args + (1, 0)
        return function(*args)

    queue = Queue()

    processes = [
        Process(
            target=lambda *args: queue.put(function(*args)),
            args=function_args + (num_workers, process_id),
        )
        for process_id in range(num_workers)
    ]

    for p in processes:
        p.start()

    def check_and_wait(process, queue):
        while True:
            try:
                done = queue.get(timeout=20.0)
                return done
            except Empty:
                if not process.is_alive():
                    return False

    for p in processes:
        done = check_and_wait(p, queue)
        if not done:
            return False

    for p in processes:
        p.join()

    return True


def run_process(function):  # , args
    if not is_supported():
        return function()

    queue = Queue()

    # wrapper = lambda *args : queue.put(function())
    wrapper = lambda: queue.put(function())

    process = Process(target=wrapper)
    process.start()
    process.join()
    result = queue.get()

    return result
