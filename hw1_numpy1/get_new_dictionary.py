"""
Задача D
Get new dictionary
В файле input_dict_name находится русско-английский словарь.
В первой строке словаря записано число слов, к которым есть перевод. Затем на каждой строке
словаря располагается слово и один или несколько переводов к нему. Слово и его переводы
разделены дефисом, переводы одного слова разделены запятой. Дефис отделён от соседних слов
пробельными символами. После запятой ставится пробельный символ.
Функция get_new_dictionary(input_dict_name, output_dict_name) должна по русско-английскому
словарю, находящемуся в input_dict_name, построить англо-русский словарь и сохранить его
в файл с именем output_dict_name в аналогичном исходному словарю формате. Словарь должен
быть полным, т.е. учитывать всю информацию, которая находилась в исходном словаре. Слова
выходного словаря должны быть отсортированы в лексикографическом порядке. Если у слова
несколько переводов - все они должны быть отсортированы в лексикографическом порядке.
Пример:
5
cat - kosha
dog - soba
good - horo, normo
bad - ploh, uzha
ugly - uzha
>>>6
>>>horo - good
>>>kosha - cat
>>>normo - good
>>>ploh - bad
>>>soba - dog
>>>uzha - bad, ugly
"""


def get_new_dictionary(input_dict_name, output_dict_name):
    output_dict = {}
    with open(input_dict_name, 'r') as inp:
        input_dict = inp.read().split('\n')[1:-1]
    for pair in input_dict:
        rus, engs = pair.split(' - ')
        for eng in engs.split(', '):
            if eng not in output_dict:
                output_dict[eng] = [rus]
            else:
                output_dict[eng] += [rus]
    with open(output_dict_name, 'w') as out:
        out_string = [str(len(output_dict))]
        for eng in sorted(output_dict):
            rus = ', '.join(sorted(output_dict[eng]))
            out_string += [' - '.join([eng, rus])]
        out.write('\n'.join(out_string))
