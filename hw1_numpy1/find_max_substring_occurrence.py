"""
Задача C
Find max substring occurrence
Написать функцию find_max_substring_occurrence(input_string), принимающую на вход непустую
строку input_string. Функция должна возвращать наибольшее число k, такое что input_string
совпадает с некоторой своей подстрокой t, выписанной k раз подряд.
Пример:
find_max_substring_occurrence('abab')
>>>2
"""
from collections import Counter


def find_max_substring_occurrence(input_string):
    max_k = min(Counter(input_string).values())
    while input_string[:len(input_string) // max_k] * max_k != input_string:
        max_k -= 1
    return max_k
