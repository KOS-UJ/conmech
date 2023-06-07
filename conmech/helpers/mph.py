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

    # wrapper = lambda *args : queue.put(function())
    wrapper = lambda: queue.put(function())

    process = Process(target=wrapper)
    process.start()
    process.join()
    result = queue.get()

    return result
