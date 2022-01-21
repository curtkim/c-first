#define FLATBUFFERS_TRACK_VERIFIER_BUFFER_SIZE

#include <iostream>
#include "monster_generated.h"  // Already includes "flatbuffers/flatbuffers.h".

using namespace MyGame::Sample;

void build(flatbuffers::FlatBufferBuilder& builder, bool sizePrefixed, const char* type_id) {
  // First, lets serialize some weapons for the Monster: A 'sword' and an 'axe'.
  flatbuffers::Offset<flatbuffers::String> weapon_one_name = builder.CreateString("Sword");
  short weapon_one_damage = 3;

  auto weapon_two_name = builder.CreateString("Axe");
  short weapon_two_damage = 5;

  // Use the `CreateWeapon` shortcut to build Weapons with all fields set.
  flatbuffers::Offset<Weapon> sword = CreateWeapon(builder, weapon_one_name, weapon_one_damage);
  flatbuffers::Offset<Weapon> axe = CreateWeapon(builder, weapon_two_name, weapon_two_damage);

  // Create a FlatBuffer's `vector` from the `std::vector`.
  std::vector<flatbuffers::Offset<Weapon>> weapons_vector;
  weapons_vector.push_back(sword);
  weapons_vector.push_back(axe);
  auto weapons = builder.CreateVector(weapons_vector);

  // Second, serialize the rest of the objects needed by the Monster.
  auto position = Vec3(1.0f, 2.0f, 3.0f);

  auto name = builder.CreateString("MyMonster");

  unsigned char inv_data[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
  auto inventory = builder.CreateVector(inv_data, 10);

  // Shortcut for creating monster with all fields set:
  auto orc = CreateMonster(builder, &position, 150, 80, name, inventory,
                           Color_Red, weapons, Equipment_Weapon, axe.Union());
  if( sizePrefixed)
    builder.FinishSizePrefixed(orc, type_id);
  else
    builder.Finish(orc, type_id);  // Serialize the root of the object.
}

void read(const Monster* monster) {
  // Get and test some scalar types from the FlatBuffer.
  assert(monster->hp() == 80);
  assert(monster->mana() == 150);  // default
  assert(monster->name()->str() == "MyMonster");

  // Get and test a field of the FlatBuffer's `struct`.
  auto pos = monster->pos();
  assert(pos);
  assert(pos->z() == 3.0f);
  (void)pos;

  // Get a test an element from the `inventory` FlatBuffer's `vector`.
  auto inv = monster->inventory();
  assert(inv);
  assert(inv->Get(9) == 9);
  (void)inv;

  // Get and test the `weapons` FlatBuffers's `vector`.
  std::string expected_weapon_names[] = { "Sword", "Axe" };
  short expected_weapon_damages[] = { 3, 5 };
  auto weps = monster->weapons();
  for (unsigned int i = 0; i < weps->size(); i++) {
    assert(weps->Get(i)->name()->str() == expected_weapon_names[i]);
    assert(weps->Get(i)->damage() == expected_weapon_damages[i]);
  }
  (void)expected_weapon_names;
  (void)expected_weapon_damages;

  // Get and test the `Equipment` union (`equipped` field).
  assert(monster->equipped_type() == Equipment_Weapon);
  auto equipped = static_cast<const Weapon *>(monster->equipped());
  assert(equipped->name()->str() == "Axe");
  assert(equipped->damage() == 5);
  (void)equipped;
}

int main(int /*argc*/, const char * /*argv*/[]) {
  assert(FLATBUFFERS_LITTLEENDIAN);

  const char* type_id = "mons";
  size_t INIT_SIZE = 160;
  // Build up a serialized buffer algorithmically:
  flatbuffers::FlatBufferBuilder builder(INIT_SIZE);

  {
    build(builder, false, type_id);
    std::cout << "size=" << builder.GetSize() << std::endl;
    // We now have a FlatBuffer we can store on disk or send over a network.

    // ** file/network code goes here :) **
    // access builder.GetBufferPointer() for builder.GetSize() bytes

    flatbuffers::Verifier verifier{builder.GetBufferPointer(), builder.GetSize()};
    assert(VerifyMonsterBuffer(verifier));
    assert(!VerifySizePrefixedMonsterBuffer(verifier));
    std::cout << "verifier.GetComputedSize() " << verifier.GetComputedSize() << std::endl;

    auto data = builder.GetBufferPointer();
    const char* identifier = flatbuffers::GetBufferIdentifier(data, false);
    std::cout << "Identifier: " << identifier << "\n";
    std::cout << "BufferHasIdentifier: " << flatbuffers::BufferHasIdentifier(data, type_id, false) << "\n";

    // Instead, we're going to access it right away (as if we just received it).
    const Monster* monster = GetMonster(data);
    read(monster);
    builder.Clear();
  }

  std::cout << "=== sizePrefixed ===\n";
  {
    const bool sizePrefixed = true;
    build(builder, sizePrefixed, type_id);
    std::cout << "size=" << builder.GetSize() << std::endl;
    // We now have a FlatBuffer we can store on disk or send over a network.

    // ** file/network code goes here :) **
    // access builder.GetBufferPointer() for builder.GetSize() bytes

    flatbuffers::Verifier verifier{builder.GetBufferPointer(), builder.GetSize()};
    assert(!VerifyMonsterBuffer(verifier));
    assert(VerifySizePrefixedMonsterBuffer(verifier));
    std::cout << "verifier.GetComputedSize() " << verifier.GetComputedSize() << std::endl;

    auto data = builder.GetBufferPointer();
    const char* identifier = flatbuffers::GetBufferIdentifier(data, sizePrefixed);
    std::cout << "Identifier: " << identifier << "\n";
    std::cout << "BufferHasIdentifier: " << flatbuffers::BufferHasIdentifier(data, type_id, sizePrefixed) << "\n";

    // Instead, we're going to access it right away (as if we just received it).
    const Monster* monster = GetSizePrefixedMonster(data);
    read(monster);
  }
}