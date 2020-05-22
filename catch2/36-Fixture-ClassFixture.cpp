// Catch has two ways to express fixtures:
// - Sections
// - Traditional class-based fixtures (this file)

// main() provided in 000-CatchMain.cpp

#include <catch2/catch.hpp>

class DBConnection
{
public:
    static DBConnection createConnection( std::string const & /*dbName*/ ) {
      return DBConnection();
    }

    bool executeSQL( std::string const & /*query*/, int const /*id*/, std::string const & arg ) {
      if ( arg.length() == 0 ) {
        throw std::logic_error("empty SQL query argument");
      }
      return true; // ok
    }
};

class UniqueTestsFixture
{
protected:
    UniqueTestsFixture()
      : conn( DBConnection::createConnection( "myDB" ) )
    {}

    int getID() {
      return ++uniqueID;
    }

protected:
    DBConnection conn;

private:
    static int uniqueID;
};

int UniqueTestsFixture::uniqueID = 0;

TEST_CASE_METHOD( UniqueTestsFixture, "Create Employee/No Name", "[create]" ) {
  REQUIRE_THROWS( conn.executeSQL( "INSERT INTO employee (id, name) VALUES (?, ?)", getID(), "") );
}

TEST_CASE_METHOD( UniqueTestsFixture, "Create Employee/Normal", "[create]" ) {
  REQUIRE( conn.executeSQL( "INSERT INTO employee (id, name) VALUES (?, ?)", getID(), "Joe Bloggs" ) );
}