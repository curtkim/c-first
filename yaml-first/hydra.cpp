#include "yaml-cpp/yaml.h"
#include <filesystem>
#include <iostream>

#include <fmt/format.h>

namespace fs = std::filesystem;

struct DbConfig {
  std::string driver;
  std::string user;
  std::string pass;
  int timeout;

  friend std::ostream &operator<<(std::ostream &os, const DbConfig &config) {
    os << "driver: " << config.driver << " user: " << config.user << " pass: " << config.pass << " timeout: "
       << config.timeout;
    return os;
  }
};

struct Field {
  std::string name;
  std::string type;

  friend std::ostream &operator<<(std::ostream &os, const Field &field) {
    os << "name: " << field.name << " type: " << field.type;
    return os;
  }

  std::string toString() {
    return "name: " + name + " type: " + type;
  }

};
struct Table {
  std::string name;
  std::vector<Field> fields;
};
struct Schema {
  std::string database;
  std::vector<Table> tables;
};

namespace YAML {
  template<>
  struct convert<DbConfig> {
    static Node encode(const DbConfig &rhs) {
      Node node;
      node["driver"] = rhs.driver;
      node["user"] = rhs.user;
      node["pass"] = rhs.pass;
      //node["timeout"] = rhs.timeout;
      return node;
    }

    static bool decode(const Node &node, DbConfig &rhs) {
      if (!node.IsMap() ) {
        return false;
      }

      rhs.driver = node["driver"].as<std::string>();
      rhs.user = node["user"].as<std::string>();
      rhs.pass = node["pass"].as<std::string>();
      //rhs.timeout = node["timeout"].as<int>();
      return true;
    }
  };

  template<>
  struct convert<Field> {
    static Node encode(const Field &rhs) {
      Node node;
      node["name"] = rhs.name;
      node["type"] = rhs.type;
      return node;

    }
    static bool decode(const Node &node, Field &rhs) {
      if (!node.IsMap() ) {
        return false;
      }

      for (YAML::const_iterator it = node.begin(); it != node.end(); ++it) {
        std::string key = it->first.as<std::string>();    // <- key
        std::string value = it->second.as<std::string>(); // <- value
        rhs.name = key;
        rhs.type = value;
      }
      return true;
    }
  };

  template<>
  struct convert<Table> {
    static Node encode(const Table &rhs) {
      Node node;
      node["name"] = rhs.name;
      for(auto field : rhs.fields)
        node["fields"].push_back(convert<Field>::encode(field));
      return node;
    }

    static bool decode(const Node &node, Table &rhs) {
      if (!node.IsMap() && !node["fields"].IsSequence()) {
        return false;
      }

      rhs.name = node["name"].as<std::string>();
      for(const Node item : node["fields"]){
        Field field;
        convert<Field>::decode(item, field);
        rhs.fields.push_back(field);
      }
      return true;
    }
  };

  template<>
  struct convert<Schema> {
    static Node encode(const Schema &rhs) {
      Node node;
      node["database"] = rhs.database;
      for(auto table : rhs.tables)
        node["tables"].push_back(convert<Table>::encode(table));
      return node;
    }

    static bool decode(const Node &node, Schema &rhs) {
      if (!node.IsMap() && !node["tables"].IsSequence()) {
        return false;
      }

      rhs.database = node["database"].as<std::string>();
      for(const Node item : node["tables"]){
        Table table;
        convert<Table>::decode(item, table);
        rhs.tables.push_back(table);
      }
      return true;
    }

  };

}


YAML::Node load(const fs::path path) {
  const fs::path parent_path = path.parent_path();

  YAML::Node root = YAML::LoadFile(path);
  YAML::Node defaults = root["defaults"];

  YAML::Node result;
  for (auto node : defaults) {
    for (YAML::const_iterator it = node.begin(); it != node.end(); ++it) {
      std::string key = it->first.as<std::string>();    // <- key
      std::string value = it->second.as<std::string>(); // <- value
      //std::cout << key << ": " << value << std::endl;

      const fs::path yaml_path = parent_path / key / (value + ".yaml");
      //std::cout << yaml_path << std::endl;
      YAML::Node loaded = YAML::LoadFile(yaml_path);
      result[key] = loaded[key];
    }
  }

  return result;
}

/*
YAML::Node& getNestedNode(YAML::Node& node, std::vector<std::string> path){
  YAML::Node& current = node;
  for(auto item : path){
    current = current[item];
    std::cout << "\t" << current << std::endl;
  }
  return current;
}
*/

int main(int argc, char **argv) {
  const fs::path path("../../hydra_conf/config.yaml");
  YAML::Node config = load(path);

  std::cout << config << std::endl;

  std::cout << "nested reference " << config["db.driver"] << std::endl;
  //std::cout << "nested reference " << getNestedNode(config, {"db", "driver"}) << std::endl;

  DbConfig dbConfig = config["db"].as<DbConfig>();
  std::cout << dbConfig << std::endl;

  Schema schema = config["schema"].as<Schema>();

  return 0;
}

