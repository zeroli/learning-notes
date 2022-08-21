#include <time.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <poll.h>

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
    struct pollfd fds[num_clients];
    for (int i = 0; i < num_clients; i++) {
        fds[i].events = POLLIN;
        fds[i].fd = client_fds[i];
    }   

    while (1) {
        int retval = poll(fds, num_clients, /*timeout=*/-1);
        if (retval < 0) {
            perror("poll_wait");
            goto fail;
        }

        int cnt = 0;
        for (int i = 0; i < num_clients; i++) {
            if (fds[i].revents & POLLIN) {
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
                cnt++;
            }
            if (fds[i].revents & POLLHUP) {
                fprintf(stdout, "[%lu][Server] client %d closed\n", time(NULL), i); 
                cnt++;
            }
        }

        if (cnt != retval) {
            fprintf(stderr,
                "[%lu][Server] Number of set file descriptors(%d) != number "
                "returned from "
                "poll_wait (%d)\n",
                time(NULL), cnt, retval);
            goto fail;
        }
    }   
fail:
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
