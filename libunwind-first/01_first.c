#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
 
#define UNW_LOCAL_ONLY
#include <libunwind.h>
 
static void print_backtrace(void)
{
    unw_context_t context;
    unw_cursor_t cursor;
    unw_word_t off, ip, sp;
    unw_proc_info_t pip;
    char procname[256];
    int ret;
 
    if (unw_getcontext(&context))
        return;
 
    if (unw_init_local(&cursor, &context))
        return;
 
    while (unw_step(&cursor) > 0) {
        if (unw_get_proc_info(&cursor, &pip))
            break;
 
        ret = unw_get_proc_name(&cursor, procname, 256, &off);
        if (ret && ret != -UNW_ENOMEM) {
            procname[0] = '?';
            procname[1] = 0;
        }
 
        unw_get_reg(&cursor, UNW_REG_IP, &ip);
        unw_get_reg(&cursor, UNW_REG_SP, &sp);
 
        fprintf(stderr, "ip = 0x%lx (%s), sp = 0x%lx\n", (long)ip, procname, (long)sp);
    }
}
 
void dummy(void)
{
    print_backtrace();
}
 
int main(void)
{
    dummy();
 
    return 0;
}

//ip = 0x55c1148ba528 (dummy), sp = 0x7ffc41cd8a10
//ip = 0x55c1148ba538 (main), sp = 0x7ffc41cd8a20
//ip = 0x7f3421cf80b3 (__libc_start_main), sp = 0x7ffc41cd8a30
//ip = 0x55c1148ba2ee (_start), sp = 0x7ffc41cd8b00