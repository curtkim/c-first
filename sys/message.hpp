//
// Created by curt on 19. 12. 24..
//

#ifndef MODERN_CPP_MESSAGE_HPP
#define MODERN_CPP_MESSAGE_HPP

#include <ostream>

// structure for message queue
struct msg_buffer {
    long msg_type;
    char msg[100];
} ;



#endif //MODERN_CPP_MESSAGE_HPP
