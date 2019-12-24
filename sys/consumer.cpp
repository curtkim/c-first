#include <stdio.h>
#include <sys/ipc.h>
#include <sys/msg.h>
// Define message queue structure
struct msg_buffer {
    long msg_type;
    char msg[100];
} message;

int main() {
    key_t my_key;
    int msg_id;
    my_key = ftok("progfile", 65); //create unique key
    msg_id = msgget(my_key, 0666 | IPC_CREAT); //create message queue and return id

    msgrcv(msg_id, &message, sizeof(message), 1, 0); //used to receive message
    // display the message
    printf("Received Message is : %s \n", message.msg);
    msgctl(msg_id, IPC_RMID, NULL); //destroy the message queue
    return 0;
}