// Copyright Axelera AI, 2026
#define _GNU_SOURCE
#include <dlfcn.h>
#include <signal.h>

/* Strip RTLD_DEEPBIND before liblsan/libasan's dlopen interceptor sees it.
 * Intel's OpenCL driver loads libigfxdbgxchg64.so with RTLD_DEEPBIND, which
 * sanitizer runtimes abort on. */
void *
dlopen(const char *filename, int flag)
{
  typedef void *(*dlopen_fn)(const char *, int);
  dlopen_fn next = dlsym(RTLD_NEXT, "dlopen");
  return next(filename, flag & ~RTLD_DEEPBIND);
}

/* Provide a STRONG sigaction at position 1 (before liblsan at position 2).
 * GCC 11's liblsan exports sigaction as WEAK; a STRONG symbol here takes
 * priority so all sigaction calls are forwarded correctly to libc rather than
 * going through liblsan's broken interceptor (whose REAL(sigaction) lookup
 * fails when the AIPU PCIe runtime is active, causing device SIGSEGV handlers
 * not to be installed and intermittent crashes). */
int
sigaction(int sig, const struct sigaction *act, struct sigaction *old)
{
  typedef int (*fn)(int, const struct sigaction *, struct sigaction *);
  static fn real = NULL;
  if (!real)
    real = dlsym(RTLD_NEXT, "sigaction");
  if (real)
    return real(sig, act, old);
  return 0;
}
