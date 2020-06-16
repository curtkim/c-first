#include <iostream>
#include <asio.hpp>

int main(int argc, char* argv[])
{
  using asio::local::stream_protocol;

  std::cout << "origin process " << ::getpid() << std::endl;
  try
  {
    asio::io_service io_service;

    stream_protocol::socket parentSocket(io_service);
    stream_protocol::socket childSocket(io_service);

    //create socket pair
    asio::local::connect_pair(childSocket, parentSocket);
    std::string request("Dad I am your child, hello!");
    std::string dadRequest("Hello son!");

    //Create child process
    pid_t pid = fork();
    if( pid < 0 ){
      std::cerr << "fork() erred\n";
    } else if (pid == 0 ) { //child process
      parentSocket.close(); // no need of parents socket handle, childSocket is bidirectional stream socket unlike pipe that has different handles for read and write
      asio::write(childSocket, asio::buffer(request)); //Send data to the parent

      std::vector<char> dadResponse(dadRequest.size(),0);
      asio::read(childSocket, asio::buffer(dadResponse)); //Wait for parents response

      std::cout << "Dads response: ";
      std::cout.write(&dadResponse[0], dadResponse.size());
      std::cout << std::endl;
      std::cout << "child process " << ::getpid() << std::endl;
    } else { //parent
      childSocket.close(); //Close childSocket here use one bidirectional socket
      std::vector<char> reply(request.size());
      asio::read(parentSocket, asio::buffer(reply)); //Wait for child process to send message

      std::cout << "Child message: ";
      std::cout.write(&reply[0], request.size());
      std::cout << std::endl;

      sleep(1); //Add 5 seconds delay before sending response to parent
      asio::write(parentSocket, asio::buffer(dadRequest)); //Send child process response
      std::cout << "dad process " << ::getpid() << std::endl;
    }
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
    std::exit(1);
  }
}
