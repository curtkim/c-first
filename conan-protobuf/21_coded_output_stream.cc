#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

int main() {
  using namespace google::protobuf::io;

  // Write some data to "myfile".  First we write a 4-byte "magic number"
  // to identify the file type, then write a length-delimited string.  The
  // string is composed of a varint giving the length followed by the raw
  // bytes.
  int fd = open("myfile", O_CREAT | O_WRONLY);
  ZeroCopyOutputStream* raw_output = new FileOutputStream(fd);
  CodedOutputStream* coded_output = new CodedOutputStream(raw_output);

  int magic_number = 1234;
  char text[] = "Hello world!";
  coded_output->WriteLittleEndian32(magic_number);
  coded_output->WriteVarint32(strlen(text));
  coded_output->WriteRaw(text, strlen(text));

  delete coded_output;
  delete raw_output;
  close(fd);
}