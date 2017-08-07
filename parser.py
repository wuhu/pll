def find_matching_bracket(string, pos=0):
    l = string.find("(", pos + 1)
    r = string.find(")", pos + 1)
    if l < 0 or l > r:
        return r
    return find_matching_bracket(string, find_matching_bracket(string, l))


def split_jumping_brackets(string, delimiter=","):
    c = string.find(delimiter)
    l = string.find("(")
    if c < 0:
        return [string]
    if l < 0:
        return string.split(delimiter)
    if c > l:
        c = string.find(delimiter, find_matching_bracket(string, l))
        if c < 0:
            return [string]
    return [string[:c]] + split_jumping_brackets(string[c + 1:], delimiter)
