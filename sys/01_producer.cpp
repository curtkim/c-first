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

    for(int i = 0; i < 3; i++){
        message.msg_type = 1;
        printf("Write Message : ");
        fgets(message.msg, 100, stdin);
        msgsnd(msg_id, &message, sizeof(message), 0); //send message
        printf("Sent message is : %s \n", message.msg);
    }

}