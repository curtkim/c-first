#include <stdio.h>
#include <sys/ipc.h>
#include <sys/msg.h>

#include "message.hpp"

int main() {
    msg_buffer message;

    key_t my_key;
    int msg_id;
    my_key = ftok("progfile", 65); //create unique key
    msg_id = msgget(my_key, 0666 | IPC_CREAT); //create message queue and return id

    for(int i = 0; i < 3; i++) {
        msgrcv(msg_id, &message, sizeof(message), 1, 0); //used to receive message
        printf("Received Message is(%d) : %s \n", message.msg_type, message.msg);
    }

    msgctl(msg_id, IPC_RMID, NULL); //destroy the message queue
    return 0;
}