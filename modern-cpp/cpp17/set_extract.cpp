#include <set>
#include <string>
#include <iostream>

struct User {
  std::string name;

  User(std::string s) : name(std::move(s)) {
    std::cout << "User::User(" << name << ")\n";
  }
  ~User() {
    std::cout << "User::~User(" << name << ")\n";
  }
  User(const User& u) : name(u.name) {
    std::cout << "User::User(copy, " << name << ")\n";
  }

  friend bool operator<(const User& u1, const User& u2) {
    return u1.name < u2.name;
  }
};

int main() {
  std::set<User> users1;
  users1.emplace("John");
  users1.emplace("Alex");

  std::set<User> users2;


  std::cout << "move John...\n";
  // move John to the outSet
  auto handle = users1.extract(User("John"));
  users2.insert(std::move(handle));

  for (auto& elem : users1)
    std::cout << elem.name << '\n';

  std::cout << "cleanup...\n";
}