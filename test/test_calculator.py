import pytest

from calculator import average, divide


def test_divide_returns_quotient() -> None:
    assert divide(10, 2) == 5


def test_divide_supports_negative_values() -> None:
    assert divide(-9, 3) == -3


def test_divide_by_zero_raises() -> None:
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)


def test_average_of_single_number() -> None:
    assert average([4]) == 4


def test_average_of_multiple_numbers() -> None:
    assert average([1, 2, 3, 4]) == 2.5


def test_average_of_negative_numbers() -> None:
    assert average([-1, -2, -3]) == -2


def test_average_of_empty_list_raises() -> None:
    with pytest.raises(ValueError):
        average([])
