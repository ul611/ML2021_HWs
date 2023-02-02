"""
Задача B
Find word in a circle
По кругу записано несколько букв (возможно, повторяющихся). Студент хочет узнать, сможет ли
он прочитать некоторое слово, если будет двигаться по кругу (в любом направлении), не
пропуская буквы.

Студент сам выбирает место, с которого он начинает читать, и направление. Необходимо
написать функцию find_word_in_circle(circle, word), которая должна возвращать кортеж из
двух элементов (если студент может найти строку word в круговой строке circle) или число -1
(если не может). Первый элемент кортежа - позиция, с которой нужно начинать чтение (индекс
в строке). Второй элемент - направление чтения (1 - слева направо, -1 - справа налево).
Строка word содержит как минимум один символ.

Если подходит несколько вариантов ответа, приоритет должен отдаваться варианту, который
проходит по часовой стрелке с наименьшим значением для начала позиции.
Пример:
find_word_in_circle('napo', 'ap')
>>>(1, 1)
"""


def find_word_in_circle(circle, word):
    if not circle:
        return -1
    full_circle = circle
    while len(full_circle) < len(word):
        full_circle += circle
    full_circle += circle[:-1:]
    pos = full_circle.find(word)
    direction = 1
    if pos > -1:
        return pos, direction
    inverted_word = word[::-1]
    pos = full_circle.find(inverted_word)
    direction = -1
    if pos > -1:
        return (pos + len(word) - 1) % len(circle), direction
    return -1
