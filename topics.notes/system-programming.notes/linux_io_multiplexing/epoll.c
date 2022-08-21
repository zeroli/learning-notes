#include <errno.h>
#include <fcntl.h>
#include <time.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/epoll.h>

#define NUM_CLIENTS 5
#define BUF_SIZE  16

static int run_client(int id, int write_fd)
{
    for (int cnt = 0; ; cnt++) {
        char buf[BUF_SIZE] = {0};
        int nbytes = snprintf(buf, sizeof(buf), "%d", cnt);
        write(write_fd, buf, nbytes);
        sleep(id + 1); 
    }   
    return 0;
}

static int run_server(int client_fds[], int num_clients)
{
    struct epoll_event epoll_events[num_clients];

    int epoll_fd = epoll_create1(/*flags=*/0);
    if (epoll_fd < 0) {
        perror("epoll_create");
        goto fail;
    }   

    for (int i = 0; i < num_clients; i++) {
        struct epoll_event event;
        event.events = EPOLLIN;
        event.data.u32 = i;

        int retval = epoll_ctl(epoll_fd, EPOLL_CTL_ADD, client_fds[i], &event);
        if (retval < 0) {
            perror("epoll_ctl");
            goto fail;
        }
    }   

    while (1) {
        int ready_cnt = epoll_wait(epoll_fd, epoll_events, num_clients, /*timeout=*/-1);
        if (ready_cnt < 0) {
            perror("epoll_wait");
            goto fail;
        }

        for (int i = 0; i < ready_cnt; i++) {
            uint32_t id = epoll_events[i].data.u32;
            if (epoll_events[i].events & EPOLLIN) {
                char buf[BUF_SIZE] = {0};
                int nbytes = read(client_fds[i], buf, sizeof(buf));
                if (nbytes < 0) {
                    perror("read");
                    goto fail;
                } else if (nbytes > 0) {
                    fprintf(stdout, "[%lu][Server] client %d: %s\n",
                        time(NULL), i, buf);
                } else {
                    fprintf(stdout, "[%lu][Server] client %d closed\n",
                        time(NULL), i); 
                }
            }
            if (epoll_events[i].events & EPOLLHUP) {
                fprintf(stdout, "[%lu][Server] client %d closed\n", time(NULL), i); 
            }
        }
    }
fail:
    if (epoll_fd > 0) {
        close(epoll_fd);
    }
    for (int i = 0; i < num_clients; i++) {
        close(client_fds[i]);
    }
}

int main()
{
    int client_fds[NUM_CLIENTS] = { 0 };

    for (int i = 0; i < NUM_CLIENTS; i++) {
        int pipefd[2] = { 0 };
        if (pipe(pipefd) != 0) {
            perror("pipe");
            return -1;
        }
        int read_fd = pipefd[0];
        int write_fd = pipefd[1];
        client_fds[i] = read_fd;

        pid_t child_pid = fork();
        if (child_pid < 0) {
            perror("fork");
            return child_pid;
        } else if (child_pid == 0) {
            close(read_fd);
            return run_client(i, write_fd);
        }
        close(write_fd);
    }
    return run_server(client_fds, NUM_CLIENTS);
}
