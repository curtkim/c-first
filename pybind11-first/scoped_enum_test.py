import scoped_enum

print(scoped_enum.ScopedEnum.Two)
print(scoped_enum.ScopedEnum.Two.name)
print(scoped_enum.ScopedEnum.Three)

a = scoped_enum.ScopedEnum.Three

if a == scoped_enum.ScopedEnum.Three:
    print('three')
else:
    print('two')

print(isinstance(scoped_enum.ScopedEnum.Three, scoped_enum.ScopedEnum))
print(scoped_enum.ScopedEnum.__members__)
print(list(scoped_enum.ScopedEnum.__members__))