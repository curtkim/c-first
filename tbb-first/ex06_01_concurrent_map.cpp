#include <tbb/concurrent_hash_map.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <string>

// Structure that defines hashing and comparison operations for user's type.
struct MyHashCompare {
  static size_t hash( const std::string& x ) {
    size_t h = 0;
    for( const char* s = x.c_str(); *s; ++s )
      h = (h*17)^*s;
    return h;
  }
  //! True if strings are equal
  static bool equal( const std::string& x, const std::string& y ) {
    return x==y;
  }
};

// A concurrent hash table that maps strings to ints.
typedef tbb::concurrent_hash_map<std::string,int,MyHashCompare> StringTable;

// Function object for counting occurrences of strings.
struct Tally {
  StringTable& table;
  Tally( StringTable& table_ ) : table(table_) {}
  void operator()( const tbb::blocked_range<std::string*> range ) const {
    // the next few lines can be improved, see Figure 6.4 (define FASTER in compilation)
    for( std::string* p=range.begin(); p!=range.end(); ++p ) {
      StringTable::accessor a;
      table.insert( a, *p );
      a->second += 1;
#ifdef FASTER
      a.release();
#endif
    }
  }
};

const size_t N = 10;

std::string Data[N] = { "Hello", "World", "TBB", "Hello",
                        "So Long", "Thanks for all the fish", "So Long",
                        "Three", "Three", "Three" };

int main() {
  // Construct empty table.
  StringTable table;

  // Put occurrences into the table
  tbb::parallel_for( tbb::blocked_range<std::string*>( Data, Data+N, 1000 ),
                     Tally(table) );

  // Display the occurrences using a simple walk
  // (note: concurrent_hash_map does not offer const_iterator)
  // see a problem with this code???
  // read "Iterating thorough these structures is asking for trouble"
  // coming up in a few pages
  for( StringTable::iterator i=table.begin(); i!=table.end(); ++i )
    printf("%s %d\n",i->first.c_str(),i->second);

  return 0;
}