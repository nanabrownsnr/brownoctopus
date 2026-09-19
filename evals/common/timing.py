from time import perf_counter


def elapsed_ms(start: float) -> float:
    return (perf_counter() - start) * 1000.0
