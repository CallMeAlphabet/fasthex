# fasthex

fasthex – a very fast hex dumper

Speed advantages:
  1. mmap path: output formatted in parallel with rayon in 64 MiB chunks.
  2. AVX2 path: processes 32 bytes (2 rows) per SIMD call; falls back to
     SSE4.1/SSSE3 (16 bytes / 1 row) or scalar.
  3. Double-buffered I/O: a dedicated writer thread drains completed chunks
     while rayon formats the next one.
  4. MADV_SEQUENTIAL on open + MADV_WILLNEED two chunks ahead to hide
     mmap page-fault latency.
  5. Zero-copy output via an internal pipe pair:
       formatted buffer
         → vmsplice  (userspace pages → kernel pipe, zero-copy)
         → splice    (kernel pipe → stdout fd, zero-copy)
     This path works regardless of what stdout is (/dev/null, file, pipe,
     socket) because we own the intermediate pipe. Falls back to write_all
     if splice rejects the stdout fd (e.g. a tty).
  6. Streaming (stdin) path uses a 4 MiB write buffer.

Row layout – 76 bytes:
  [0..7]   8-digit hex offset
  [8]      ' '
  [9..32]  hex group-1  (8 bytes × "XX ")
  [33]     extra gap space
  [34..57] hex group-2  (8 bytes × "XX ")
  [58]     ' '
  [59..74] ASCII printable / '.'
  [75]     '\n'

NOTE: the offset field is u32, so files larger than 4 GiB will have a
wrapping offset display. Known limitation.
