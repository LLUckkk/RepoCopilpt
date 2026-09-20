def divide(a: float, b: float) -> float:
    return a / b


def average(numbers: list[float]) -> float:
    if not numbers:
        return 0.0
    total = sum(numbers)
    return total / len(numbers)


print(average([]))