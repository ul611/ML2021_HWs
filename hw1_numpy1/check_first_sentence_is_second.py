"""Задача A
Check first sentence is second
Напишите функцию check_first_sentence_is_second, принимающую на вход две строки.
Каждая строка задаёт предложение. Необходимо проверить, можно ли получить второе
предложения из первого с помощью перестановки и удаления слов. Каждая из строк может
содержать только буквы и пробелы, любые буквенные последовательности разделённые
пробелом считаются разными словами. Если можно, функция должна вернуть True, иначе False.
Пример:
check_first_sentence_is_second("люк я твой отец", "я отец твой")
>>>True
"""
from collections import Counter


def check_first_sentence_is_second(sentence_1, sentence_2):
    dict_1 = Counter(sentence_1.split())
    dict_2 = Counter(sentence_2.split())
    answer = True
    for word in dict_2:
        if word not in dict_1 or dict_1[word] < dict_2[word]:
            answer = False
            break
    return answer
