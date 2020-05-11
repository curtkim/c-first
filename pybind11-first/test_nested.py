import nested
import nested.util

a = nested.Person()
a.name = "curt"
a.age = 40
print(a.name)

print(nested.util.format_person(a))
