# fasthex

fasthex – a very fast hex dumper (written in Rust)

## Benchmarks (790 MiB file)

| Tool | Time | Speed vs fasthex |
|------|------|------------------|
| **fasthex** | **0.57s** | **1x (baseline)** |
| xxd | 16.84s | 30x slower |
| hexyl | 40.34s | 71x slower |
| hexdump | 53.67s | 95x slower |

*Tested on i5-7500T with file not being in RAM disk redirected to /dev/null.*
*All commands were executed without any additional flags.*

## How to set it up

1. Install `cargo` (and optionally `time`)
2. Clone this repo (It'll put everything into `~/fasthex` automatically)
```bash
git clone https://github.com/CallMeAlphabet/fasthex
```
3. Compile, put into `~/.local/bin`
```bash
cd ~/fasthex && cargo build --release && cp ~/fasthex/target/release/fasthex ~/.local/bin/ && cd ~
```
Or, if `~/.local/bin` isn't in PATH and you don't want to put `~/.local/bin` in PATH
```bash
cp ~/fasthex/target/release/fasthex /usr/bin && cd ~
```
4. Clean up
```bash
rm -rf ~/fasthex
```
5. Test it
```bash
# Get help:
fasthex -h

# Normal test
time fasthex ~/path/to/file

# Pipe it (faster, vmsplice, kernel pipe)
time fasthex ~/path/to/file > /dev/null

# Put into RAM (slow SSDs / HDDs can bottleneck a lot)
sudo mkdir -p /mnt/ramdisk
sudo mount -t tmpfs -o size=[MAKE SURE YOUR FILE FITS] tmpfs /mnt/ramdisk                                                                                                              cp ~/path/to/file /mnt/ramdisk/
time fasthex /mnt/ramdisk/file > /dev/null
```
NOTE: If you need `sudo`, you may need to use the full path to `fasthex`, if `fasthex` is in `~/.local/bin`

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

Standard row layout – 76 bytes:

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

```bash
◄ 0s ◎ fasthex -h                                                                                                                                                       
Usage:
 fasthex [options] <file>.

Options:
 -b, --one-byte-octal      one-byte octal display
 -X, --one-byte-hex        one-byte hexadecimal display
 -c, --one-byte-char       one-byte character display
 -d, --two-bytes-decimal   two-byte decimal display
 -o, --two-bytes-octal     two-byte octal display
 -x, --two-bytes-hex       two-byte hexadecimal display
 -L, --color[=<hexValue>]  color the output
 -n, --length <length>     interpret only length bytes of input
 -s, --skip <offset>       skip offset bytes from the beginning
 -w, --with-squeezing      do not output identical lines
 -i, --no-ascii            do not display ascii
 -a, --binary              binary display
 -v, --version             display version (always outputs 2.0)
 -h, --help                display this help page

Arguments:
 Values for <length> and <offset> may be followed by a suffix: KiB, MiB,
 GiB, TiB, PiB, EiB, ZiB, or YiB (where the "iB" is optional).
                                                                                                                                                                               
◄ 0s ◎ echo "This repo is not actively maintained as `fasthex` is fast enough and I have other projects. Maybe some features will be added in the future! And yes, I had to put this into an 'echo "[...]"', because I couldn't close the code block for some reason. And it looked really weird putting that note into a code block. So I decided that I should echo it. I'm weird."
