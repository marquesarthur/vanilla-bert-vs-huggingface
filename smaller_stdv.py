x = [0.47, 0.40, 0.43, 0.49]
y = [0.39, 0.36, 0.48, 0.49]


_min_std_precision = 50.
_min_std_recall = 50.


print("precision")
np.std(x)
print("recall")
np.std(y)


_min_std_precision = min(_min_std_precision, np.std(x))
_min_std_recall = min(_min_std_recall, np.std(y))



x = [0.47, 0.60, 0.53, 0.50]
y = [0.40, 0.54, 0.55, 0.45]

print("precision")
np.std(x)
print("recall")
np.std(y)

_min_std_precision = min(_min_std_precision, np.std(x))
_min_std_recall = min(_min_std_recall, np.std(y))


x = [0.53, 0.44, 0.49, 0.50]
y = [0.52, 0.36, 0.45, 0.47]

print("precision")
np.std(x)
print("recall")
np.std(y)


_min_std_precision = min(_min_std_precision, np.std(x))
_min_std_recall = min(_min_std_recall, np.std(y))


print("\n\MIN Precision")
print(_min_std_precision)
print("MIN recall")
print(_min_std_recall)


print("\n\n")
print("-" * 10)


# ----


_min_std_precision = 50.
_min_std_recall = 50.

x = [0.52, 0.52, 0.58, 0.52]
y = [0.55, 0.56, 0.63, 0.56]

print("precision")
np.std(x)
print("recall")
np.std(y)


_min_std_precision = min(_min_std_precision, np.std(x))
_min_std_recall = min(_min_std_recall, np.std(y))

x = [0.51, 0.53, 0.53, 0.51]
y = [0.57, 0.55, 0.58, 0.54]

print("precision")
np.std(x)
print("recall")
np.std(y)

_min_std_precision = min(_min_std_precision, np.std(x))
_min_std_recall = min(_min_std_recall, np.std(y))


x = [0.52, 0.52, 0.58, 0.53]
y = [0.55, 0.56, 0.56, 0.57]

print("precision")
np.std(x)
print("recall")
np.std(y)



_min_std_precision = min(_min_std_precision, np.std(x))
_min_std_recall = min(_min_std_recall, np.std(y))


print("\n\MIN Precision")
print(_min_std_precision)
print("MIN recall")
print(_min_std_recall)


print("\n\n")
print("-" * 10)