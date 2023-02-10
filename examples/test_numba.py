import sys

import numba
import numpy as np

from conmech.helpers.tmh import Timer

## Numba tests


def find_instr(func, keyword, sig=0, limit=5):
    count = 0
    for l in func.inspect_asm(func.signatures[sig]).split("\n"):
        if keyword in l:
            count += 1
            print(l)
            if count >= limit:
                break
    if count == 0:
        print("No instructions found")
    print(count)


def save_numba_llvm(fun, out):
    original_stdout = sys.stdout
    with open(f"{out}/{fun.__name__}.ll", "w") as file:
        sys.stdout = file
        for v, k in fun.inspect_llvm().items():
            print(v, k)
        sys.stdout = original_stdout


def save_numba_asm(fun, out):
    original_stdout = sys.stdout
    with open(f"{out}/{fun.__name__}.s", "w") as file:
        sys.stdout = file
        for v, k in fun.inspect_asm().items():
            print(v, k)
        sys.stdout = original_stdout


def save_numba_types(fun, out):
    original_stdout = sys.stdout
    with open(f"{out}/{fun.__name__}.txt", "w") as file:
        sys.stdout = file
        print(iterate_test_loop.inspect_types())
    sys.stdout = original_stdout


def save_numba_all(fun, out):
    save_numba_types(fun=fun, out=out)
    save_numba_llvm(fun=fun, out=out)
    save_numba_asm(fun=fun, out=out)

###

@numba.njit()
def iterate_test_loop(values):
    c = 0
    mask = np.empty_like(values)
    for i in range(len(values)):
        for j in range(len(values)):
            x = values[j] - values[i]
            c += 1
    return c


@numba.njit()
def iterate_test_broadcast(values):
    c = 0
    for i in range(len(values)):
        mask = values - values[i]
        for j in range(len(values)):
            a = mask[i]
            c += 1
    return c


# @numba.njit()
# def iterate_test_loop(values):
#     for i in range(len(values)):
#         for j in range(len(values)):
#             values[j] += 1
#     return values


# @numba.njit()
# def iterate_test_loop2(values):
#     for i in range(len(values)):
#         for j in range(len(values)):
#             values[i] += 1
#     return values


# @numba.njit()
# def iterate_test_broadcast(values):
#     for j in range(len(values)):
#         values += 1
#     return values


values = np.zeros((100, 3))

timer = Timer()

with timer["iterate_test_broadcast"]:
    result_broadcast = iterate_test_broadcast(values)

with timer["iterate_test_loop"]:
    result_loop = iterate_test_loop(values)

print(timer.to_dataframe())






out = "output"
save_numba_all(fun=iterate_test_loop, out=out)
save_numba_all(fun=iterate_test_broadcast, out=out)


#   store <2 x i64>
#   store <4 x i64>

# SSE: xmmX
# AVX/AVX2: ymmX
# AVX-512: zmmX
