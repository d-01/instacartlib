def function_A():
    raise Exception('function_A has been called')


def function_B():
    raise Exception('function_B has been called')


class ClassA:
    pass


exports = {
    'function_A': function_A,
    'function_B': function_B,
    'ClassA': ClassA,
}