#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <fstream>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

int main() {
  using namespace google::protobuf::io;

  // Read a file created by the above code.
  int fd = open("myfile", O_RDONLY);
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);

  unsigned int magic_number;
  coded_input->ReadLittleEndian32(&magic_number);

  if (magic_number != 1234) {
    std::cerr << "File not in expected format." << std::endl;
    return -1;
  }

  unsigned int size;
  coded_input->ReadVarint32(&size);

  char* text = new char[size + 1];
  coded_input->ReadRaw(text, size);
  text[size] = '\0';

  delete coded_input;
  delete raw_input;
  close(fd);

  std::cout << "Text is: " << text << std::endl;
  delete [] text;
}