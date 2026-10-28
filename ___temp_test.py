words = ["apple", "banana", "orange"]
doc = "apple banana orange"

result = (
    1 for word in words
    if word in doc
)

print(next(result))
print(next(result))
print(next(result))

print(result)