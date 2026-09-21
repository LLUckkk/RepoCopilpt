def divide(a: float, b: float) -> float:
    return a / b


def average(numbers: list[float]) -> float:
    if not numbers:
        raise ValueError("average() requires at least one number")
    total = sum(numbers)
    return total / len(numbers)


print(average([1, 2, 3, 4]))