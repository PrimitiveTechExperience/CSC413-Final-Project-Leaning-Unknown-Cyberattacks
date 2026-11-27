# Test prediction examples

Windows are decoded system-call sequences (runs of the same call are shown as `call×n`; at most 30 groups). Probabilities are P(malicious) from each completed run; a window is flagged when its probability is at least that run's threshold τ, chosen on validation data.

## How different are benign and malicious test windows?

Total variation (TV) distance between the system-call frequency distributions of benign and malicious test windows (0 = identical mix of calls, 1 = no calls in common).

| Scenario | TV distance | Calls most over-represented in malicious windows |
|---|---|---|
| Bruteforce_CWE-307 | 0.255 | `read` (20.2% vs 12.5%), `poll` (9.7% vs 6.0%), `writev` (8.9% vs 6.2%) |
| CVE-2012-2122 | 0.172 | `fcntl` (23.1% vs 21.0%), `shutdown` (1.4% vs 0.0%), `procexit` (1.4% vs 0.0%) |
| CVE-2014-0160 | 0.016 | `writev` (6.8% vs 6.3%), `mmap` (3.2% vs 2.9%), `semop` (4.2% vs 4.0%) |
| CVE-2017-7529 | 0.248 | `stat` (6.3% vs 0.2%), `accept` (6.4% vs 1.5%), `epoll_ctl` (6.5% vs 1.8%) |
| CVE-2018-3760 | 0.106 | `newfstatat` (22.5% vs 14.0%), `read` (5.0% vs 4.0%), `mprotect` (1.7% vs 1.0%) |
| CVE-2019-5418 | 0.046 | `futex` (48.6% vs 44.9%), `getpid` (3.9% vs 3.3%), `write` (3.2% vs 3.0%) |
| CVE-2020-9484 | 0.053 | `sched_yield` (13.6% vs 11.4%), `futex` (28.6% vs 27.2%), `mmap` (0.4% vs 0.1%) |
| CWE-89-SQL-injection | 0.047 | `recvfrom` (14.9% vs 13.5%), `poll` (11.3% vs 10.4%), `sendto` (6.2% vs 5.7%) |
| PHP_CWE-434 | 0.050 | `munmap` (3.6% vs 2.9%), `io_getevents` (1.7% vs 1.0%), `mmap` (1.9% vs 1.4%) |
| EPS_CWE-434 | 0.018 | `read` (22.7% vs 22.0%), `brk` (10.7% vs 10.2%), `rt_sigaction` (0.8% vs 0.7%) |
| ZipSlip | 0.037 | `gettid` (20.6% vs 19.1%), `futex` (22.0% vs 21.2%), `sysinfo` (3.6% vs 3.3%) |
| CVE-2017-12635_6 | 0.886 | `close` (91.0% vs 2.4%), `rt_sigaction` (0.0% vs 0.0%), `brk` (0.0% vs 0.0%) |
| CVE-2020-13942 | 0.074 | `lstat` (20.9% vs 19.1%), `futex` (50.8% vs 49.5%), `epoll_wait` (3.0% vs 1.8%) |
| CVE-2020-23839 | 0.090 | `stat` (3.7% vs 2.1%), `getdents64` (3.7% vs 2.7%), `fstat` (10.3% vs 9.6%) |
| Juice-Shop | 0.007 | `futex` (31.4% vs 31.0%), `getpid` (6.8% vs 6.6%), `madvise` (0.2% vs 0.1%) |

## Bruteforce_CWE-307

9,132 test windows, 1,370 malicious. Examples picked with the Transformer runs.

**Highest-scoring attack window when held out (still below τ: no held-out detections)** (malicious, 25 calls)

P(malicious): LSTM Seen 0.553, LSTM Seen (all) 0.995*, LSTM Held-out 0.624, Transformer Seen 0.997*, Transformer Seen (all) 0.999*, Transformer Held-out 0.783 (* = flagged)

`wait4 select wait4 select wait4 select wait4 select wait4 select wait4 select clone wait4 select wait4 select wait4 select wait4 select wait4 select wait4 select`

**Missed when held out but caught when seen (held-out false negative)** (malicious, 100 calls)

P(malicious): LSTM Seen 1.000*, LSTM Seen (all) 1.000*, LSTM Held-out 0.215, Transformer Seen 1.000*, Transformer Seen (all) 1.000*, Transformer Held-out 0.153 (* = flagged)

`writev shutdown close read semop epoll_wait accept semop getsockname open read close fcntl×2 read writev poll read writev poll read stat open×3 fstat read×2 close write writev write times shutdown …`

**Missed in every condition (false negative, lowest seen (all) score)** (malicious, 200 calls)

P(malicious): LSTM Seen 0.140, LSTM Seen (all) 0.099, LSTM Held-out 0.003, Transformer Seen 0.074, Transformer Seen (all) 0.082, Transformer Held-out 0.013 (* = flagged)

`setitimer open fstat×4 mmap munmap close rename umask×2 chmod chdir umask setitimer×2 brk×2 writev write times poll read writev shutdown close read semop epoll_wait accept semop getsockname open read close …`

**Highest-scoring benign window when held out (still below τ: no false alarms)** (benign, 50 calls)

P(malicious): LSTM Seen 0.448, LSTM Seen (all) 0.345, LSTM Held-out 0.581, Transformer Seen 0.206, Transformer Seen (all) 0.552, Transformer Held-out 0.689 (* = flagged)

`epoll_wait accept semop getsockname open read close mmap×2 fcntl×2 mmap brk mmap read writev poll read writev poll read stat open×3 mmap fstat read close write mmap×2 poll read×2 brk …`

**Typical benign window (true negative)** (benign, 50 calls)

P(malicious): LSTM Seen 0.205, LSTM Seen (all) 0.140, LSTM Held-out 0.006, Transformer Seen 0.165, Transformer Seen (all) 0.178, Transformer Held-out 0.017 (* = flagged)

`fstat read close setitimer rt_sigaction rt_sigprocmask poll read×2 getcwd open write×2 close umask×2 getcwd chdir setitimer open fstat×4 mmap munmap close rename umask×2 chmod chdir umask setitimer×2 brk×2 writev write …`


## CVE-2012-2122

36,188 test windows, 26,574 malicious. Examples picked with the Transformer runs.

**Detected without seeing this attack (held-out true positive, median score)** (malicious, 34 calls)

P(malicious): LSTM Seen 1.000*, LSTM Seen (all) 1.000*, LSTM Held-out 0.039, Transformer Seen 1.000*, Transformer Seen (all) 1.000*, Transformer Held-out 0.978* (* = flagged)

`set_robust_list setsockopt×2 getpeername setsockopt write read fcntl read×2 fcntl write shutdown close madvise exit procexit set_robust_list setsockopt×2 getpeername setsockopt write read fcntl read×2 fcntl write shutdown close madvise exit procexit`

**Missed when held out but caught when seen (held-out false negative)** (malicious, 17 calls)

P(malicious): LSTM Seen 1.000*, LSTM Seen (all) 1.000*, LSTM Held-out 0.059, Transformer Seen 1.000*, Transformer Seen (all) 1.000*, Transformer Held-out 0.901 (* = flagged)

`set_robust_list setsockopt×2 getpeername setsockopt write read fcntl read×2 fcntl write shutdown close madvise exit procexit`

**Missed in every condition (false negative, lowest seen (all) score)** (malicious, 100 calls)

P(malicious): LSTM Seen 0.006, LSTM Seen (all) 0.040, LSTM Held-out 0.000, Transformer Seen 0.007, Transformer Seen (all) 0.026, Transformer Held-out 0.008 (* = flagged)

`fcntl×2 accept fcntl getsockname fcntl×2 setsockopt×2 fcntl setsockopt×2 mmap mprotect clone poll fcntl×2 accept fcntl getsockname fcntl×2 setsockopt×2 fcntl setsockopt×2 mmap mprotect clone poll fcntl×2 accept fcntl getsockname fcntl×2 setsockopt×2 …`

**Highest-scoring benign window when held out (still below τ: no false alarms)** (benign, 8 calls)

P(malicious): LSTM Seen 0.019, LSTM Seen (all) 0.440, LSTM Held-out 0.153, Transformer Seen 0.081, Transformer Seen (all) 0.506, Transformer Held-out 0.623 (* = flagged)

`futex×2 select×6`

**Typical benign window (true negative)** (benign, 100 calls)

P(malicious): LSTM Seen 0.362, LSTM Seen (all) 0.599, LSTM Held-out 0.000, Transformer Seen 0.119, Transformer Seen (all) 0.813, Transformer Held-out 0.000 (* = flagged)

`read fcntl setsockopt write setsockopt read fcntl read×2 fcntl setsockopt write setsockopt read fcntl read×2 fcntl setsockopt pwrite fsync write setsockopt read fcntl read×2 fcntl setsockopt mprotect×2 write setsockopt read …`


## CVE-2014-0160

2,433 test windows, 175 malicious. Examples picked with the Transformer runs.

**Highest-scoring attack window when held out (still below τ: no held-out detections)** (malicious, 100 calls)

P(malicious): LSTM Seen 0.997*, LSTM Seen (all) 0.999*, LSTM Held-out 0.039, Transformer Seen 0.616, Transformer Seen (all) 0.179, Transformer Held-out 0.522 (* = flagged)

`poll read writev shutdown close read semop epoll_wait accept semop getsockname open read close fcntl×2 read writev poll read writev×4 poll read close read semop epoll_wait accept semop getsockname open …`

**Missed when held out but caught when seen (held-out false negative)** (malicious, 27 calls)

P(malicious): LSTM Seen 0.998*, LSTM Seen (all) 1.000*, LSTM Held-out 0.001, Transformer Seen 0.994*, Transformer Seen (all) 0.943*, Transformer Held-out 0.070 (* = flagged)

`epoll_wait accept semop getsockname open read close mmap×2 fcntl×2 mmap brk mmap read writev poll read writev×4 poll read close read semop`

**Missed in every condition (false negative, lowest seen (all) score)** (malicious, 200 calls)

P(malicious): LSTM Seen 0.314, LSTM Seen (all) 0.160, LSTM Held-out 0.011, Transformer Seen 0.427, Transformer Seen (all) 0.080, Transformer Held-out 0.023 (* = flagged)

`epoll_wait accept semop getsockname open read close mmap×2 fcntl×2 mmap brk mmap read writev poll read writev poll read stat open×3 mmap fstat read close mmap setitimer rt_sigaction rt_sigprocmask mmap …`

**False alarm when held out (benign, highest score)** (benign, 200 calls)

P(malicious): LSTM Seen 0.375, LSTM Seen (all) 0.184, LSTM Held-out 0.013, Transformer Seen 0.351, Transformer Seen (all) 0.909, Transformer Held-out 0.970* (* = flagged)

`mmap munmap mmap writev write times poll read writev shutdown close read semop epoll_wait accept semop getsockname open read close fcntl×2 read writev poll read poll read writev poll read …`

**Typical benign window (true negative)** (benign, 11 calls)

P(malicious): LSTM Seen 0.601, LSTM Seen (all) 0.360, LSTM Held-out 0.002, Transformer Seen 0.740, Transformer Seen (all) 0.405, Transformer Held-out 0.113 (* = flagged)

`fcntl read writev poll read writev poll read close read semop`


## CVE-2017-7529

1,462 test windows, 328 malicious. Examples picked with the Transformer runs.

**Highest-scoring attack window when held out (still below τ: no held-out detections)** (malicious, 5 calls)

P(malicious): LSTM Seen 0.997*, LSTM Seen (all) 0.869, LSTM Held-out 0.296, Transformer Seen 0.999*, Transformer Seen (all) 0.471, Transformer Held-out 0.386 (* = flagged)

`write×2 close×2 epoll_wait`

**Missed when held out but caught when seen (held-out false negative)** (malicious, 100 calls)

P(malicious): LSTM Seen 1.000*, LSTM Seen (all) 1.000*, LSTM Held-out 0.011, Transformer Seen 0.999*, Transformer Seen (all) 1.000*, Transformer Held-out 0.002 (* = flagged)

`epoll_wait accept epoll_ctl epoll_wait recvfrom stat open fstat writev sendfile write×2 close×2 epoll_wait accept epoll_ctl epoll_wait recvfrom stat open fstat writev sendfile write×2 close×2 epoll_wait accept epoll_ctl epoll_wait recvfrom stat …`

**Missed in every condition (false negative, lowest seen (all) score)** (malicious, 200 calls)

P(malicious): LSTM Seen 0.049, LSTM Seen (all) 0.065, LSTM Held-out 0.001, Transformer Seen 0.066, Transformer Seen (all) 0.023, Transformer Held-out 0.003 (* = flagged)

`recvfrom open write writev write recvfrom epoll_wait accept epoll_ctl epoll_wait recvfrom stat open fstat writev sendfile write close setsockopt recvfrom epoll_wait accept epoll_ctl epoll_wait accept epoll_ctl epoll_wait accept epoll_ctl epoll_wait …`

**Highest-scoring benign window when held out (still below τ: no false alarms)** (benign, 8 calls)

P(malicious): LSTM Seen 0.269, LSTM Seen (all) 0.622, LSTM Held-out 0.273, Transformer Seen 0.061, Transformer Seen (all) 0.220, Transformer Held-out 0.624 (* = flagged)

`epoll_wait recvfrom open write writev write recvfrom epoll_wait`

**Typical benign window (true negative)** (benign, 100 calls)

P(malicious): LSTM Seen 0.040, LSTM Seen (all) 0.049, LSTM Held-out 0.000, Transformer Seen 0.060, Transformer Seen (all) 0.035, Transformer Held-out 0.002 (* = flagged)

`sendfile write close recvfrom epoll_wait recvfrom open fstat writev sendfile write close recvfrom epoll_wait recvfrom open write writev write recvfrom epoll_wait accept epoll_ctl epoll_wait recvfrom stat open fstat writev sendfile …`


## CVE-2018-3760

50,000 test windows, 1,053 malicious. Examples picked with the Transformer runs.

**Highest-scoring attack window when held out (still below τ: no held-out detections)** (malicious, 4 calls)

P(malicious): LSTM Seen 0.962*, LSTM Seen (all) 0.314, LSTM Held-out 0.320, Transformer Seen 0.841*, Transformer Seen (all) 0.506, Transformer Held-out 0.505 (* = flagged)

`select×4`

**Missed when held out but caught when seen (held-out false negative)** (malicious, 50 calls)

P(malicious): LSTM Seen 0.985*, LSTM Seen (all) 0.936*, LSTM Held-out 0.030, Transformer Seen 0.823*, Transformer Seen (all) 0.967*, Transformer Held-out 0.030 (* = flagged)

`mprotect read×2 close stat open ioctl read mprotect read close stat open ioctl mprotect read×2 close stat open ioctl read×2 close stat open ioctl mprotect read×2 close stat open ioctl …`

**Missed in every condition (false negative, lowest seen (all) score)** (malicious, 50 calls)

P(malicious): LSTM Seen 0.351, LSTM Seen (all) 0.099, LSTM Held-out 0.009, Transformer Seen 0.354, Transformer Seen (all) 0.011, Transformer Held-out 0.001 (* = flagged)

`stat×50`

**False alarm when held out (benign, highest score)** (benign, 100 calls)

P(malicious): LSTM Seen 0.334, LSTM Seen (all) 0.113, LSTM Held-out 0.005, Transformer Seen 0.364, Transformer Seen (all) 0.152, Transformer Held-out 0.988* (* = flagged)

`stat×51 setsockopt futex write×2 setsockopt getpid select write futex×2 getpid×2 write futex×9 select getpid×2 write futex×2 recvfrom getsockname getsockopt getpeername stat×3 getpid getrandom getpid stat×2 getpid×2 openat fstat fcntl×2 getdents×2 …`

**Typical benign window (true negative)** (benign, 100 calls)

P(malicious): LSTM Seen 0.244, LSTM Seen (all) 0.096, LSTM Held-out 0.009, Transformer Seen 0.420, Transformer Seen (all) 0.083, Transformer Held-out 0.009 (* = flagged)

`close utimensat stat open ioctl fstat lseek read×2 close utimensat getpid write×2 getpid setsockopt write×6 setsockopt getpid select write futex×3 getsockopt getpid open×69`


## CVE-2019-5418

50,000 test windows, 1,512 malicious. Examples picked with the Transformer runs.

**Highest-scoring attack window when held out (still below τ: no held-out detections)** (malicious, 30 calls)

P(malicious): LSTM Seen 0.712*, LSTM Seen (all) 0.204, LSTM Held-out 0.208, Transformer Seen 0.639*, Transformer Seen (all) 0.630, Transformer Held-out 0.518 (* = flagged)

`select getpid×2 write futex×8 getpid×2 write futex×4 select futex×3 select futex×4 select×2`

**Missed in every condition (false negative, lowest seen (all) score)** (malicious, 100 calls)

P(malicious): LSTM Seen 0.169, LSTM Seen (all) 0.077, LSTM Held-out 0.036, Transformer Seen 0.490, Transformer Seen (all) 0.035, Transformer Held-out 0.060 (* = flagged)

`futex getpid×2 write futex×5 write getpid×2 write futex×3 write getpid×2 write futex×3 mprotect×8 getrandom×3 futex stat getpid×2 write futex×4 open fstat mprotect getpid×2 write futex×2 lseek getdents×2 close futex stat …`

**False alarm when held out (benign, highest score)** (benign, 200 calls)

P(malicious): LSTM Seen 0.423, LSTM Seen (all) 0.133, LSTM Held-out 0.111, Transformer Seen 0.504, Transformer Seen (all) 0.074, Transformer Held-out 0.999* (* = flagged)

`newfstatat×155 stat open ioctl fstat lseek read×2 close stat newfstatat×32 mprotect newfstatat×3`

**Typical benign window (true negative)** (benign, 200 calls)

P(malicious): LSTM Seen 0.546, LSTM Seen (all) 0.172, LSTM Held-out 0.002, Transformer Seen 0.573*, Transformer Seen (all) 0.129, Transformer Held-out 0.010 (* = flagged)

`futex×4 select recvfrom getsockname getsockopt getpeername futex getpid×2 write futex×4 stat futex×2 stat getpid×2 write futex×6 stat getpid×2 write futex×4 getpid getrandom stat futex write getpid×2 write futex×6 stat futex …`


## CVE-2020-9484

50,000 test windows, 2,257 malicious. Examples picked with the Transformer runs.

**Detected without seeing this attack (held-out true positive, median score)** (malicious, 50 calls)

P(malicious): LSTM Seen 0.991*, LSTM Seen (all) 0.998*, LSTM Held-out 0.998*, Transformer Seen 0.991*, Transformer Seen (all) 0.999*, Transformer Held-out 0.999* (* = flagged)

`close×50`

**Missed when held out but caught when seen (held-out false negative)** (malicious, 50 calls)

P(malicious): LSTM Seen 0.991*, LSTM Seen (all) 1.000*, LSTM Held-out 0.916, Transformer Seen 0.991*, Transformer Seen (all) 0.998*, Transformer Held-out 0.123 (* = flagged)

`getpid rt_sigprocmask rt_sigaction×3 setpgid rt_sigprocmask ioctl rt_sigprocmask close read close rt_sigaction×19 execve brk access openat stat openat stat openat stat openat stat openat stat openat stat openat stat openat stat`

**Missed in every condition (false negative, lowest seen (all) score)** (malicious, 100 calls)

P(malicious): LSTM Seen 0.489, LSTM Seen (all) 0.153, LSTM Held-out 0.003, Transformer Seen 0.446, Transformer Seen (all) 0.026, Transformer Held-out 0.000 (* = flagged)

`futex×2 lseek read lseek read futex×3 lseek read lseek read lseek read lseek read lseek read lseek read futex×4 write futex dup futex×2 read futex×12 dup futex×2 read futex×4 stat×2 …`

**Highest-scoring benign window when held out (still below τ: no false alarms)** (benign, 200 calls)

P(malicious): LSTM Seen 0.437, LSTM Seen (all) 0.424, LSTM Held-out 0.009, Transformer Seen 0.667*, Transformer Seen (all) 0.993*, Transformer Held-out 0.664 (* = flagged)

`lstat openat dup fstat fcntl×2 getdents64 lstat×106 stat×2 openat fstat fcntl×2 read lseek×2 read mmap stat openat fstat fcntl×2 read lseek×2 read mmap stat openat fstat fcntl×2 read lseek×2 read …`

**Typical benign window (true negative)** (benign, 100 calls)

P(malicious): LSTM Seen 0.466, LSTM Seen (all) 0.188, LSTM Held-out 0.000, Transformer Seen 0.454, Transformer Seen (all) 0.219, Transformer Held-out 0.000 (* = flagged)

`epoll_wait futex epoll_ctl epoll_wait read epoll_ctl epoll_wait futex epoll_ctl epoll_wait read epoll_ctl epoll_wait futex epoll_ctl epoll_wait close read epoll_ctl epoll_wait futex epoll_ctl epoll_wait read epoll_ctl epoll_wait futex×2 epoll_ctl×2 epoll_wait futex×2 …`


## CWE-89-SQL-injection

50,000 test windows, 6,448 malicious. Examples picked with the Transformer runs.

**Detected without seeing this attack (held-out true positive, median score)** (malicious, 26 calls)

P(malicious): LSTM Seen 0.564, LSTM Seen (all) 0.988*, LSTM Held-out 0.579, Transformer Seen 0.999*, Transformer Seen (all) 0.999*, Transformer Held-out 0.999* (* = flagged)

`io_getevents×26`

**Missed when held out but caught when seen (held-out false negative)** (malicious, 50 calls)

P(malicious): LSTM Seen 0.999*, LSTM Seen (all) 1.000*, LSTM Held-out 0.121, Transformer Seen 0.998*, Transformer Seen (all) 0.967*, Transformer Held-out 0.081 (* = flagged)

`socket fcntl×2 connect poll getsockopt fcntl setsockopt×2 poll recvfrom poll recvfrom sendto poll recvfrom poll recvfrom sendto poll recvfrom poll recvfrom poll recvfrom sendto close chdir setitimer utime close sendto …`

**Missed in every condition (false negative, lowest seen (all) score)** (malicious, 88 calls)

P(malicious): LSTM Seen 0.383, LSTM Seen (all) 0.198, LSTM Held-out 0.348, Transformer Seen 0.023, Transformer Seen (all) 0.002, Transformer Held-out 0.010 (* = flagged)

`io_getevents×18 futex io_getevents×69`

**False alarm when held out (benign, highest score)** (benign, 32 calls)

P(malicious): LSTM Seen 0.528, LSTM Seen (all) 0.771, LSTM Held-out 0.507, Transformer Seen 0.996*, Transformer Seen (all) 0.999*, Transformer Held-out 0.998* (* = flagged)

`io_getevents×32`

**Typical benign window (true negative)** (benign, 100 calls)

P(malicious): LSTM Seen 0.306, LSTM Seen (all) 0.260, LSTM Held-out 0.080, Transformer Seen 0.391, Transformer Seen (all) 0.271, Transformer Held-out 0.070 (* = flagged)

`lstat×5 stat lstat×6 stat setitimer rt_sigaction rt_sigprocmask poll read getcwd lstat getpid open write×2 close getcwd chdir setitimer fcntl getcwd open fstat getuid flock fcntl fstat pread access getcwd×3 open …`


## PHP_CWE-434

50,000 test windows, 1,971 malicious. Examples picked with the Transformer runs.

**Detected without seeing this attack (held-out true positive, median score)** (malicious, 32 calls)

P(malicious): LSTM Seen 0.979*, LSTM Seen (all) 0.993*, LSTM Held-out 0.999*, Transformer Seen 0.992*, Transformer Seen (all) 0.994*, Transformer Held-out 0.973* (* = flagged)

`execve brk access×2 open fstat mmap close access open read fstat mmap×2 mprotect mmap×2 close arch_prctl mprotect×3 munmap brk×2 uname fstat write close×2 exit_group procexit`

**Missed when held out but caught when seen (held-out false negative)** (malicious, 200 calls)

P(malicious): LSTM Seen 0.942*, LSTM Seen (all) 1.000*, LSTM Held-out 0.219, Transformer Seen 0.989*, Transformer Seen (all) 0.999*, Transformer Held-out 0.283 (* = flagged)

`execve brk access×2 open fstat mmap close access open read fstat mmap×2 mprotect mmap close access open read fstat mmap mprotect mmap close access open read fstat mmap mprotect mmap …`

**Missed in every condition (false negative, lowest seen (all) score)** (malicious, 50 calls)

P(malicious): LSTM Seen 0.433, LSTM Seen (all) 0.147, LSTM Held-out 0.074, Transformer Seen 0.466, Transformer Seen (all) 0.070, Transformer Held-out 0.044 (* = flagged)

`set_robust_list rt_sigaction×3 getpid×2 mmap×2 geteuid setgid open read close open lseek fstat mmap lseek fstat munmap close setgroups geteuid setuid getpid×2 epoll_create1 epoll_ctl mmap accept getsockname fcntl×2 mmap×2 read stat …`

**False alarm when held out (benign, highest score)** (benign, 50 calls)

P(malicious): LSTM Seen 0.439, LSTM Seen (all) 0.967*, LSTM Held-out 0.993*, Transformer Seen 0.407, Transformer Seen (all) 0.936*, Transformer Held-out 0.999* (* = flagged)

`getsockname fcntl×2 read stat×2 lstat×6 stat lstat×6 stat lstat×6 stat setitimer rt_sigaction rt_sigprocmask getcwd chdir setitimer fcntl getcwd open fstat getuid flock fcntl fstat pread access getcwd×3 open lseek fstat …`

**Typical benign window (true negative)** (benign, 200 calls)

P(malicious): LSTM Seen 0.465, LSTM Seen (all) 0.234, LSTM Held-out 0.170, Transformer Seen 0.435, Transformer Seen (all) 0.270, Transformer Held-out 0.192 (* = flagged)

`lstat×2 stat lstat×6 stat lstat×6 stat setitimer rt_sigaction rt_sigprocmask getcwd chdir setitimer fcntl getcwd open fstat getuid flock fcntl fstat pread access getcwd×3 open lseek fstat mmap lseek munmap close …`


## EPS_CWE-434

50,000 test windows, 1,047 malicious. Examples picked with the Transformer runs.

**Detected without seeing this attack (held-out true positive, median score)** (malicious, 200 calls)

P(malicious): LSTM Seen 0.559*, LSTM Seen (all) 1.000*, LSTM Held-out 1.000*, Transformer Seen 0.731*, Transformer Seen (all) 0.987*, Transformer Held-out 0.985* (* = flagged)

`close access open read fstat mmap mprotect mmap close access open read fstat mmap×2 mprotect mmap×2 close access open read fstat mmap mprotect mmap×2 close access open read fstat mmap …`

**Missed when held out but caught when seen (held-out false negative)** (malicious, 200 calls)

P(malicious): LSTM Seen 0.944*, LSTM Seen (all) 1.000*, LSTM Held-out 1.000*, Transformer Seen 0.919*, Transformer Seen (all) 0.972*, Transformer Held-out 0.225 (* = flagged)

`mprotect munmap set_tid_address set_robust_list rt_sigaction×2 rt_sigprocmask getrlimit brk×2 clock_getres×2 stat open fstat read×3 close stat ioctl×2 rt_sigaction×5 stat open fstat mmap close futex statfs stat×2 open write open fstat read×4 …`

**Missed in every condition (false negative, lowest seen (all) score)** (malicious, 22 calls)

P(malicious): LSTM Seen 0.535, LSTM Seen (all) 0.081, LSTM Held-out 0.000, Transformer Seen 0.534, Transformer Seen (all) 0.016, Transformer Held-out 0.000 (* = flagged)

`brk×4 munmap×2 fstat write close munmap brk munmap close munmap×5 brk close exit_group procexit`

**False alarm when held out (benign, highest score)** (benign, 16 calls)

P(malicious): LSTM Seen 0.537, LSTM Seen (all) 0.563, LSTM Held-out 0.457, Transformer Seen 0.536, Transformer Seen (all) 0.233, Transformer Held-out 0.967* (* = flagged)

`execve brk access×2 open fstat mmap close access open read fstat mmap×2 mprotect mmap`

**Typical benign window (true negative)** (benign, 96 calls)

P(malicious): LSTM Seen 0.534, LSTM Seen (all) 0.105, LSTM Held-out 0.000, Transformer Seen 0.571, Transformer Seen (all) 0.093, Transformer Held-out 0.000 (* = flagged)

`open×3 close open×3 close open close open read×2 close stat umask open umask open write×12 stat umask open umask write×7 writev×10 write close open×14 rt_sigaction×2 rt_sigprocmask clone wait4 rt_sigaction×2 rt_sigprocmask …`


## ZipSlip

50,000 test windows, 1,940 malicious. Examples picked with the Transformer runs.

**Highest-scoring attack window when held out (still below τ: no held-out detections)** (malicious, 50 calls)

P(malicious): LSTM Seen 0.488, LSTM Seen (all) 0.318, LSTM Held-out 0.330, Transformer Seen 0.504, Transformer Seen (all) 0.272, Transformer Held-out 0.261 (* = flagged)

`futex×50`

**Missed when held out but caught when seen (held-out false negative)** (malicious, 100 calls)

P(malicious): LSTM Seen 0.517, LSTM Seen (all) 0.994*, LSTM Held-out 0.004, Transformer Seen 0.986*, Transformer Seen (all) 0.990*, Transformer Held-out 0.046 (* = flagged)

`close access open read fstat mmap mprotect mmap close open×2 access open read fstat mmap mprotect mmap×2 close mprotect×2 munmap open×3 fstat mmap close access open read fstat mmap mprotect …`

**Missed in every condition (false negative, lowest seen (all) score)** (malicious, 200 calls)

P(malicious): LSTM Seen 0.516, LSTM Seen (all) 0.119, LSTM Held-out 0.000, Transformer Seen 0.471, Transformer Seen (all) 0.007, Transformer Held-out 0.000 (* = flagged)

`futex×70 sysinfo futex×2 gettid×7 futex sysinfo futex×77 gettid×8 mprotect×7 futex×2 mprotect×5 futex×2 sysinfo futex×16`

**Highest-scoring benign window when held out (still below τ: no false alarms)** (benign, 29 calls)

P(malicious): LSTM Seen 0.774*, LSTM Seen (all) 0.444, LSTM Held-out 0.186, Transformer Seen 0.772*, Transformer Seen (all) 0.511, Transformer Held-out 0.721 (* = flagged)

`sendto wait4 pipe clone close read close wait4 shutdown close poll×3 accept recvfrom stat open fstat ioctl lseek recvfrom close wait4 shutdown close poll×4`

**Typical benign window (true negative)** (benign, 50 calls)

P(malicious): LSTM Seen 0.526, LSTM Seen (all) 0.221, LSTM Held-out 0.000, Transformer Seen 0.532, Transformer Seen (all) 0.210, Transformer Held-out 0.000 (* = flagged)

`gettid×3 mprotect gettid×9 mprotect gettid mprotect gettid×3 mprotect gettid×2 mprotect×25 futex mprotect×2`


## CVE-2017-12635_6

50,000 test windows, 21,944 malicious. Examples picked with the Transformer runs.

**Detected without seeing this attack (held-out true positive, median score)** (malicious, 100 calls)

P(malicious): LSTM Seen 0.991*, LSTM Seen (all) 0.998*, LSTM Held-out 1.000*, Transformer Seen 0.993*, Transformer Seen (all) 0.999*, Transformer Held-out 1.000* (* = flagged)

`close×100`

**Missed when held out but caught when seen (held-out false negative)** (malicious, 200 calls)

P(malicious): LSTM Seen 0.991*, LSTM Seen (all) 0.997*, LSTM Held-out 0.987*, Transformer Seen 0.992*, Transformer Seen (all) 0.985*, Transformer Held-out 0.634 (* = flagged)

`execve brk access mmap access open fstat mmap close access open read fstat mmap mprotect mmap×2 close mmap×3 arch_prctl mprotect×2 munmap dup×2 close×173`

**Missed in every condition (false negative, lowest seen (all) score)** (malicious, 50 calls)

P(malicious): LSTM Seen 0.988*, LSTM Seen (all) 0.988*, LSTM Held-out 0.185, Transformer Seen 0.966*, Transformer Seen (all) 0.103, Transformer Held-out 0.110 (* = flagged)

`rt_sigprocmask rt_sigaction×22 execve brk access mmap access open fstat mmap close access open read fstat mmap mprotect mmap×2 close access open read fstat mmap mprotect mmap close access`

**False alarm when held out (benign, highest score)** (benign, 200 calls)

P(malicious): LSTM Seen 0.991*, LSTM Seen (all) 0.998*, LSTM Held-out 1.000*, Transformer Seen 0.993*, Transformer Seen (all) 0.999*, Transformer Held-out 1.000* (* = flagged)

`close×200`

**Typical benign window (true negative)** (benign, 50 calls)

P(malicious): LSTM Seen 0.082, LSTM Seen (all) 0.266, LSTM Held-out 0.001, Transformer Seen 0.083, Transformer Seen (all) 0.254, Transformer Held-out 0.000 (* = flagged)

`sched_yield×32 futex sched_yield×17`


## CVE-2020-13942

50,000 test windows, 1,534 malicious. Examples picked with the Transformer runs.

**Detected without seeing this attack (held-out true positive, median score)** (malicious, 50 calls)

P(malicious): LSTM Seen 0.987*, LSTM Seen (all) 0.998*, LSTM Held-out 0.998*, Transformer Seen 0.919*, Transformer Seen (all) 0.999*, Transformer Held-out 0.999* (* = flagged)

`close×50`

**Missed when held out but caught when seen (held-out false negative)** (malicious, 19 calls)

P(malicious): LSTM Seen 0.912*, LSTM Seen (all) 0.960*, LSTM Held-out 0.268, Transformer Seen 0.972*, Transformer Seen (all) 0.944*, Transformer Held-out 0.299 (* = flagged)

`epoll_wait×19`

**Missed in every condition (false negative, lowest seen (all) score)** (malicious, 100 calls)

P(malicious): LSTM Seen 0.281, LSTM Seen (all) 0.075, LSTM Held-out 0.014, Transformer Seen 0.085, Transformer Seen (all) 0.004, Transformer Held-out 0.008 (* = flagged)

`lseek read futex×56 lseek read lseek read futex×38`

**Highest-scoring benign window when held out (still below τ: no false alarms)** (benign, 12 calls)

P(malicious): LSTM Seen 0.459, LSTM Seen (all) 0.329, LSTM Held-out 0.223, Transformer Seen 0.626, Transformer Seen (all) 0.739, Transformer Held-out 0.574 (* = flagged)

`futex mprotect×4 futex×2 mprotect×4 futex`

**Typical benign window (true negative)** (benign, 200 calls)

P(malicious): LSTM Seen 0.282, LSTM Seen (all) 0.032, LSTM Held-out 0.004, Transformer Seen 0.074, Transformer Seen (all) 0.014, Transformer Held-out 0.012 (* = flagged)

`futex×3 epoll_ctl epoll_wait close read futex×2 epoll_wait read epoll_wait futex×3 read epoll_ctl epoll_wait futex read futex×17 lseek read futex×28 lseek read futex×15 lseek read lseek read futex×9 lseek read lseek …`


## CVE-2020-23839

36,911 test windows, 1,267 malicious. Examples picked with the Transformer runs.

**Detected without seeing this attack (held-out true positive, median score)** (malicious, 39 calls)

P(malicious): LSTM Seen 0.999*, LSTM Seen (all) 1.000*, LSTM Held-out 0.998*, Transformer Seen 0.997*, Transformer Seen (all) 0.998*, Transformer Held-out 1.000* (* = flagged)

`execve brk access openat fstat mmap close openat read fstat mmap×2 mprotect mmap×4 close arch_prctl mprotect×3 munmap brk×2 fstat openat fstat fadvise64 mmap read write read munmap close×3 exit_group procexit`

**Missed when held out but caught when seen (held-out false negative)** (malicious, 200 calls)

P(malicious): LSTM Seen 0.999*, LSTM Seen (all) 1.000*, LSTM Held-out 0.000, Transformer Seen 0.996*, Transformer Seen (all) 0.998*, Transformer Held-out 0.000 (* = flagged)

`access×5 openat fstat lseek fstat read×3 close openat fstat lseek fstat read×3 close access×2 chdir setitimer fcntl read writev write times poll read stat openat×2 stat rt_sigaction×7 setitimer rt_sigaction rt_sigprocmask …`

**Missed in every condition (false negative, lowest seen (all) score)** (malicious, 50 calls)

P(malicious): LSTM Seen 0.008, LSTM Seen (all) 0.003, LSTM Held-out 0.001, Transformer Seen 0.022, Transformer Seen (all) 0.005, Transformer Held-out 0.000 (* = flagged)

`set_robust_list rt_sigaction×4 getpid×2 mmap×2 geteuid setgid openat read close openat lseek fstat read lseek×32`

**False alarm when held out (benign, highest score)** (benign, 19 calls)

P(malicious): LSTM Seen 0.934, LSTM Seen (all) 0.803, LSTM Held-out 0.168, Transformer Seen 0.992*, Transformer Seen (all) 0.996*, Transformer Held-out 0.999* (* = flagged)

`fstat read×3 close chdir setitimer fcntl read writev write times poll shutdown poll read close read accept`

**Typical benign window (true negative)** (benign, 200 calls)

P(malicious): LSTM Seen 0.025, LSTM Seen (all) 0.021, LSTM Held-out 0.000, Transformer Seen 0.111, Transformer Seen (all) 0.018, Transformer Held-out 0.000 (* = flagged)

`chdir setitimer fcntl access getcwd access openat fstat lseek fstat read×3 close openat fstat getdents64×2 close access×2 openat fstat lseek fstat read×3 close access×6 openat fstat getdents64×2 close access openat …`


## Juice-Shop

50,000 test windows, 2,408 malicious. Examples picked with the Transformer runs.

**Highest-scoring attack window when held out (still below τ: no held-out detections)** (malicious, 3 calls)

P(malicious): LSTM Seen 0.494*, LSTM Seen (all) 0.485, LSTM Held-out 0.335, Transformer Seen 0.457, Transformer Seen (all) 0.439, Transformer Held-out 0.530 (* = flagged)

`futex write futex`

**Missed in every condition (false negative, lowest seen (all) score)** (malicious, 100 calls)

P(malicious): LSTM Seen 0.489, LSTM Seen (all) 0.003, LSTM Held-out 0.000, Transformer Seen 0.512*, Transformer Seen (all) 0.054, Transformer Held-out 0.000 (* = flagged)

`getpid×75 futex madvise×24`

**Highest-scoring benign window when held out (still below τ: no false alarms)** (benign, 4 calls)

P(malicious): LSTM Seen 0.495*, LSTM Seen (all) 0.389, LSTM Held-out 0.337, Transformer Seen 0.466, Transformer Seen (all) 0.442, Transformer Held-out 0.519 (* = flagged)

`write futex×3`

**Typical benign window (true negative)** (benign, 200 calls)

P(malicious): LSTM Seen 0.495*, LSTM Seen (all) 0.224, LSTM Held-out 0.000, Transformer Seen 0.500*, Transformer Seen (all) 0.195, Transformer Held-out 0.000 (* = flagged)

`open fcntl futex×2 open fcntl open fcntl open fcntl open fcntl futex×4 open fcntl open fcntl futex×4 open fcntl open fcntl open fcntl open fcntl open fcntl futex×3 fcntl×3 stat …`
