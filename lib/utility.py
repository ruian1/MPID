import time


def timestr():
    t0 = time.time() + 60 * 60 * 2
    return time.strftime("%Y%m%d-%I_%M_%p",time.localtime(t0))

