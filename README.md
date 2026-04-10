# fasthex

fasthex – a very fast hex dumper (written in Rust)

## Benchmarks (790 MiB file)

| Tool | Time | Speed vs fasthex |
|------|------|------------------|
| **fasthex** | **0.57s** | **1x (baseline)** |
| xxd | 16.84s | 30x slower |
| hexyl | 40.34s | 71x slower |
| hexdump | 53.67s | 95x slower |

*Tested on i5-7500T with file (not in RAM) redirected to /dev/null.*
*All commands were executed without any additional flags.*

## How to set it up

1. Install `cargo` (and optionally `time`)
2. Clone this repo (It'll put everything into `~/fasthex` automatically):
```bash
git clone https://github.com/CallMeAlphabet/fasthex
```
3. Compile, put into `~/.local/bin`
```bash
cd ~/fasthex && cargo build --release && cp ~/fasthex/target/release/fasthex ~/.local/bin/fasthex && cd ~
```
4. Clean up
```bash
rm -rf ~/fasthex
```
5. Test it
```bash
# Normal test
time fasthex ~/path/to/file

# Pipe it (faster, vmsplice, kernel pipe)
time fasthex ~/path/to/file > /dev/null

# Put into RAM (slow SSDs / HDDs can bottleneck a lot)
sudo mkdir -p /mnt/ramdisk
sudo mount -t tmpfs -o size=[MAKE SURE YOUR FILE FITS] tmpfs /mnt/ramdisk                                                                                                              cp ~/path/to/file /mnt/ramdisk/
time fasthex /mnt/ramdisk/file > /dev/null
```
NOTE: If you need `sudo`, you may need to use the full path to `fasthex`.

## Speed advantages:
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

`  [0..7]   8-digit hex offset`

`  [8]      ' '`
  
`  [9..32]  hex group-1  (8 bytes × "XX ")`
  
`  [33]     extra gap space`
  
`  [34..57] hex group-2  (8 bytes × "XX ")`
  
`  [58]     ' '`
  
`  [59..74] ASCII printable / '.'`
  
`  [75]     '\n'`

NOTE: the offset field is u32, so files larger than 4 GiB will have a
wrapping offset display. Known limitation.


This repo is not actively maintained as (1) `fasthex` is fast enough and (2) I have other projects and (3) this was just an experiment. 
