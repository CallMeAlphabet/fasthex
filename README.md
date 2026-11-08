# fasthex

fasthex — a very fast hex dumper (written in Rust), with all features that other hex dumpers have too.

## Table of Contents

- [Quick Start](#quick-start)
- [Benchmarks](#benchmarks)
- [Uninstall](#uninstall)
- [Usage](#usage)
- [Features](#features)
- [Example Output](#example-output)
- [How It Works](#how-it-works)
- [Testing Conditions](#testing-conditions)

## Quick Start
- **On Arch**
```bash
paru -S fasthex
```
- **Non-arch**
```bash
cargo install fasthex
```
> **Note**: Make sure `~/.cargo/bin` is in your `PATH`. It's added automatically by rustup, but if `fasthex` isn't found, add this to your shell config file:
> ```bash
> # If you use Bash:
> export PATH="$HOME/.cargo/bin:$PATH"
>
> # If you use Fish:
> fish_add_path $HOME/.cargo/bin
>
> # If you use Zsh:
> export PATH="$PATH:$HOME/.cargo/bin"
> ```

```bash
fasthex /path/to/file
```

## Benchmarks

| File | Mode | Tool | Mean wall time | fasthex speedup |
|---|---|---|---:|---:|
| **random-5GiB.bin** | **canonical** | **fasthex 0.3.7** | **3.937 ± 0.451 s** | **—** |
| random-5GiB.bin | canonical | HastyHex | 22.958 ± 0.602 s | 5.8× |
| **fasthex-16MiB.bin** | **canonical** | **fasthex 0.3.7** | **21.4 ± 1.1 ms** | **—** |
| fasthex-16MiB.bin | canonical | HastyHex | 68.5 ± 1.3 ms | 3.2× |
| fasthex-16MiB.bin | canonical | hexyl | 555.5 ± 18.6 ms | 25.9× |
| fasthex-16MiB.bin | canonical | xxd | 709.6 ± 20.5 ms | 33.1× |
| fasthex-16MiB.bin | canonical | hexdump | 3367.8 ± 117.6 ms | 157.1× |
| **fasthex-16MiB.bin** | **plain hex** | **fasthex 0.3.7** | **26.5 ± 6.4 ms** | **—** |
| fasthex-16MiB.bin | plain hex | xxd | 226.0 ± 3.2 ms | 8.5× |

Reproduce with Hyperfine:

```bash
hyperfine --shell=default --warmup 0 --runs 2 \
  --prepare 'cat "$HOME/benchmark-inputs/random-5GiB.bin" > /dev/null' \
  './target/release/fasthex "$HOME/benchmark-inputs/random-5GiB.bin" > /dev/null' \
  '/tmp/hastyhex/hastyhex -p -f < "$HOME/benchmark-inputs/random-5GiB.bin" > /dev/null'

hyperfine --shell=default --warmup 1 --runs 3 \
  './target/release/fasthex /tmp/fasthex-16MiB.bin > /dev/null' \
  'xxd -g 1 -c 16 /tmp/fasthex-16MiB.bin > /dev/null' \
  'hexdump -v -C /tmp/fasthex-16MiB.bin > /dev/null' \
  'hexyl --color=never --border=none --character-table=ascii -v /tmp/fasthex-16MiB.bin > /dev/null' \
  '/tmp/hastyhex/hastyhex -p -f < /tmp/fasthex-16MiB.bin > /dev/null'

hyperfine --shell=default --warmup 1 --runs 10 \
  './target/release/fasthex -p /tmp/fasthex-16MiB.bin > /dev/null' \
  'xxd -p -c 0 /tmp/fasthex-16MiB.bin > /dev/null'
```

## Uninstall

```bash
cargo uninstall fasthex
```

## Usage

### Basic usage

```bash
fasthex /path/to/file

fasthex /path/to/file > output.txt

fasthex -s 1KiB -n 512 /path/to/file

fasthex --color=always /path/to/file

fasthex -b /path/to/file
```

### Common use cases

```bash
sudo mkdir -p /mnt/ramdisk
sudo mount -t tmpfs -o size=2G tmpfs /mnt/ramdisk
cp /path/to/file /mnt/ramdisk
time fasthex /mnt/ramdisk/file > /dev/null

time fasthex /path/to/file

fasthex /path/to/file | less
```

## Features

### Supported Output Formats

- Octal (`-o`)
- Decimal (`-d`)
- Binary (`-b`)
- Colored output (`--color=always`)
- ... and more!

### Display Options

- ASCII representation (default, disable with `-A`)
- Line squeezing for identical lines (`-w`)
- Arbitrary skip/length with size suffixes (`KiB, MiB, GiB, TiB, PiB, EiB, ZiB, YiB`)

## Example Output

```
❯ fasthex3 /bin/ls | head
00000000: 7f 45 4c 46 02 01 01 00  00 00 00 00 00 00 00 00  |.ELF............|
00000010: 03 00 3e 00 01 00 00 00  f0 54 00 00 00 00 00 00  |..>......T......|
00000020: 40 00 00 00 00 00 00 00  a8 73 02 00 00 00 00 00  |@........s......|
00000030: 00 00 00 00 40 00 38 00  0e 00 40 00 1c 00 1b 00  |....@.8...@.....|
00000040: 06 00 00 00 04 00 00 00  40 00 00 00 00 00 00 00  |@.......@.......|
00000050: 40 00 00 00 00 00 00 00  f0 54 00 00 00 00 00 00  |@...T...........|
00000060: 10 03 00 00 00 00 00 00  10 03 00 00 00 00 00 00  |................|
00000070: 08 00 00 00 00 00 00 00  03 00 00 00 04 00 00 00  |................|
00000080: 74 03 00 00 00 00 00 00  74 03 00 00 00 00 00 00  |t.......t.......|
00000090: 74 03 00 00 00 00 00 00  1c 00 00 00 00 00 00 00  |t...............|
```

## How It Works:
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

### Full Help

```
fasthex 0.3.7 - a very fast hex dumper

Usage:
  fasthex [options] [file]...
  fasthex -r [options] [file] [-j <offset>]
  fasthex [options] -          read from stdin explicitly

Multiple files are concatenated and treated as one stream.
If no file is given, reads from stdin.

OUTPUT FORMAT
  Rule: lowercase = one-byte mode, UPPERCASE = two-byte mode.

      (default)               canonical hex + ASCII display
  -x, --hex                   one-byte hexadecimal display
  -X, --hex-wide              two-byte hexadecimal display
  -o, --octal                 one-byte octal display
  -O, --octal-wide            two-byte octal display
  -d, --decimal               one-byte decimal display
  -D, --decimal-wide          two-byte decimal display
  -c, --chars                 one-byte character display
  -b, --binary                binary display (8 bits per byte)
  -p, --plain                 plain hex string, no offset or ASCII
  -i, --include               C include file style output
  -r, --reverse               convert hex dump back to binary

LAYOUT
  -W, --width <N>             bytes per row (default: 16)
  -g, --group <N>             bytes per group: 1, 2, 4, 8
  -E, --endian <MODE>         big | little  (default: big)
  -B, --border <STYLE>        none | ascii | unicode  (default: none)
  -A, --no-ascii              hide the ASCII panel
  -P, --no-position           hide the offset/position column
      --minimal               compact rows: offset + hex + ascii, no separators

OFFSET & NAVIGATION
  -s, --skip <N>              skip first N bytes (negative = from end)
  -n, --length <N>            read only N bytes
  -j, --jump <N>              bias added to every displayed offset
  -u, --uppercase             uppercase hex digits (A-F)
      --offset-dec            show offsets in decimal

COLOR
  -L, --color <WHEN>          auto | always | never  (default: auto)
  -S, --scheme <NAME>         default | type | gradient
  -T, --table <MODE>          ascii | default | braille | cp437 | ebcdic

FILTERING & FLOW
  -w, --squeeze               replace identical rows with '*'
  -m, --max-lines <N>         stop after N output lines
  -q, --quiet                 suppress warnings

CUSTOM FORMAT
  -F, --format <FMT>          hexdump -e style format string
  -f, --format-file <FILE>    read format strings from file

MISC
  -h, --help                  show this help
  -v, --version               show version

SIZE SUFFIXES: KiB/K/MiB/M/GiB/G/TiB/T/PiB/P/EiB/E  kB/MB/GB/TB/PB/EB  0x…
```

## Testing Conditions

https://gist.github.com/CallMeAlphabet/4b7022c4b1a8849e6943526de6a23582
