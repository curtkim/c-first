import nested2
import nested2.format1
import nested2.format2

a = nested2.Person()
a.name = "curt"
a.age = 40
print(a.name)

print(nested2.format1.format_person(a))
print(nested2.format2.format_person(a))
