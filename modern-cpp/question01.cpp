#include <stdio.h>
#include <deque>
#include <string>
#include <memory>
#include <cstring>
#include <iostream>

class chat_message
{
public:
  enum { header_length = 4 };
  enum { max_body_length = 512 };

  chat_message(): body_length_(0)
  {
    std::cout << "chat_message con" << std::endl;
  }

  chat_message(const chat_message & that) {
    std::cout << "chat_message copy con" << std::endl;
    std::memcpy(this->body(), that.data(), that.body_length_);
    this->body_length_ = that.body_length_;
  }

  ~chat_message() {
    std::cout << "chat_message decon " << body() << " data addr:" << &data_ << std::endl;
  }

  const char* data() const
  {
    return data_;
  }

  char* data()
  {
    return data_;
  }

  std::size_t length() const
  {
    return header_length + body_length_;
  }

  const char* body() const
  {
    return data_ + header_length;
  }

  char* body()
  {
    return data_ + header_length;
  }

  std::size_t body_length() const
  {
    return body_length_;
  }

  void body_length(std::size_t new_length)
  {
    body_length_ = new_length;
    if (body_length_ > max_body_length)
      body_length_ = max_body_length;
  }

  bool decode_header()
  {
    char header[header_length + 1] = "";
    std::strncat(header, data_, header_length);
    body_length_ = std::atoi(header);
    if (body_length_ > max_body_length)
    {
      body_length_ = 0;
      return false;
    }
    return true;
  }

  void encode_header()
  {
    char header[header_length + 1] = "";
    std::sprintf(header, "%4d", static_cast<int>(body_length_));
    std::memcpy(data_, header, header_length);
  }

private:
  char data_[header_length + max_body_length];
  std::size_t body_length_;
};


void * operator new(size_t size)
{
  std::cout << "New operator overloading " << std::endl;
  void * p = malloc(size);
  return p;
}

void operator delete(void * p)
{
  std::cout << "Delete operator overloading " << std::endl;
  free(p);
}

void write(chat_message& msg){
  std::cout << msg.length() << " data=" << msg.data() << " addr=" << &msg << std::endl;

}

int main() {

  std::cout << sizeof(chat_message) << std::endl;
  //chat_message* pMsg = new chat_message;


  char line[chat_message::max_body_length + 1];
  while (std::cin.getline(line, chat_message::max_body_length + 1)) {
    if( std::strlen(line) == 0)
      break;

    chat_message msg;
    msg.body_length(std::strlen(line));
    std::memcpy(msg.body(), line, msg.body_length());
    msg.encode_header();

    std::cout << "--- addr=" << &msg<< std::endl;

    write(msg);
  }

  //delete pMsg;
}