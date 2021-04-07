#include <refl-cpp/refl.hpp>

#include <cassert>

// refl-cpp proxies intecept calls to T's members
// and statically determine the member descriptor and
// type of the user-provided handler.

// An example of a generic user-defined builder-style factory
template <typename T>
class builder : public refl::runtime::proxy<builder<T>, T> {
public:

  template <typename... Args>
  builder(Args&&... args)
    : value_(std::forward<Args>(args)...)
  {
  }

  // Intercepts calls to T's members with
  // a mutable *this and a single argument
  template <typename Member, typename Value>
  static builder& invoke_impl(builder& self, Value&& value)
  {
    // Create instance of statically-determined member
    // descriptor to use helpers with ADL-lookup
    constexpr Member member;
    // Statically verify that the target member is writable
    static_assert(is_writable(member));
    // Set the value of the target field
    member(self.value_) = std::forward<Value>(value);
    // Return reference to builder
    return self;
  }

  T build()
  {
    return std::move(value_);
  }

private:
  T value_; // Backing object
};

struct User
{
  const long id;
  std::string email;
  std::string first_name;
  std::string last_name;

  User(long id)
    : id{id}
    , email{}
    , first_name{}
    , last_name{}
  {
  }
};

REFL_AUTO
(
  type(User),
  field(id),
  field(email),
  field(first_name),
  field(last_name)
)

// Metadata available at compile-time (erased at runtime)
// -> zero-cost introspection in C++17 🔥
static_assert(refl::reflect<User>().members.size == 4);

int main()
{
  // User-defined builder-style factories for any reflectable type! 🔥
  const User user = builder<User>(10)
    // .id(42) <- Fails at compile-time (is_writable == false)
    .email("jdoe@example.com")
    .first_name("John")
    .last_name("Doe")
    .build();

  assert(user.email == "jdoe@example.com");
}