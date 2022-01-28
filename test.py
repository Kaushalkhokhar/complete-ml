import os

print(os.path.dirname(__file__))

input_str = "aaasssdddfffggggaa"

count = 1
str_ = input_str[0]
for i, ch in enumerate(input_str[1:]):
    if ch == str_[-1]:
        count += 1
        if i == len(input_str)-2:
            slice = str_[-1]
            slice = str(count) + slice
            str_ = str_[:-1] + slice
    else:
        slice = str_[-1]
        slice = str(count) + slice
        str_ = str_[:-1] + slice
        str_ += ch
        count = 1

# count = 1
# str_ = input_str[0]
# for i, ch in enumerate(input_str[1:]):
#     str_ += ch
    

print(str_)