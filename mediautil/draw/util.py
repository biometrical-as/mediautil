from collections import defaultdict
from random import randint


def get_random_color_map(
    a_min: int = 0,
    a_max: int = 255,
    n: int = 3,
) -> defaultdict:
    """
    Returns a defaultdict that defaults to a tuple  of n random values between a_min and a_max
    :param a_min: Min array value
    :param a_max: Max array value
    :param n: Number of random values
    :return: Defaultdict with values
    """
    return defaultdict(lambda: tuple(int(randint(a_min, a_max)) for _ in range(n)))
