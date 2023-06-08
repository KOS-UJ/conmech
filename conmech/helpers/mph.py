"""
multiprocessing helpers
"""
import sys
from multiprocessing import Process, Queue


def is_supported():
    return "linux" in sys.platform


if not is_supported():
    print("Warning: Multiprocessing implemented only for Linux")


def run_process(function):  # , args
    if not is_supported():
        return function()

    queue = Queue()

    process = Process(target=lambda: queue.put(function()))
    process.start()
    process.join()
    result = queue.get()

    return result
