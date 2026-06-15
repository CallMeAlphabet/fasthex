# fasthex

fasthex – a very fast hex dumper (written in Rust), with all features that other hexdumpers have too.

## Table of Contents

- [Quick Start](#quick-start)
- [Benchmarks](#benchmarks)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Example Output](#example-output)
- [How It Works](#how-it-works)
- [Known Limitations](#known-limitations)
- [Testing Conditions](#testing-conditions)

## Quick Start

```bash
# Install
cargo install --git https://github.com/CallMeAlphabet/fasthex
```

> **Note**: Make sure `~/.cargo/bin` is in your `PATH`. It's added automatically by rustup, but if `fasthex` isn't found, add this to your shell config:
> ```bash
> export PATH="$HOME/.cargo/bin:$PATH"
> ```

```bash
# Use it!
fasthex /path/to/file
```

## Benchmarks

### Benchmark 1: 1.5 GiB file (output to /dev/null)

| Tool | Time | Speed vs fasthex |
|------|------|------------------|
| **fasthex** | **0.78s** | **1x (baseline)** |
| xxd | 43.62s | 55.5x slower |
| hexyl¹ | 53.14s | 67.6x slower |
| hexyl | 104.93s | 133.5x slower |
| hexdump | 116.03s | 147.5x slower |

**¹ *--color=never was used to avoid colors.***

### Benchmark 2: 69 GiB file (output to /dev/null)

| Tool | Time |
|------|------|
| fasthex | ~1m |
| hexyl | ~1h 30m |

## Installation

### Prerequisites

- Rust and Cargo

### Steps

1. Install fasthex
```bash
cargo install --git https://github.com/CallMeAlphabet/fasthex
```

2. Verify installation
```bash
fasthex -h
```

## Uninstall

```bash
cargo uninstall fasthex
```

## Usage

### Basic usage

```bash
# Display a file
fasthex /path/to/file

# Pipe it (faster with zero-copy output)
fasthex /path/to/file > output.txt

# Skip first 1 KiB and read only 512 bytes
fasthex -s 1KiB -n 512 /path/to/file

# Display with colors
fasthex -L /path/to/file

# Binary display
fasthex -a /path/to/file
```

### Common use cases

```bash
# Test on RAM disk (avoids SSD/HDD bottleneck)
sudo mkdir -p /mnt/ramdisk
sudo mount -t tmpfs -o size=2G tmpfs /mnt/ramdisk
cp /path/to/file /mnt/ramdisk
time fasthex /mnt/ramdisk/file > /dev/null

# Measure time with `time`
time fasthex /path/to/file

# Interactive viewing
fasthex /path/to/file | less
```

## Features

### Supported Output Formats

- Hexadecimal (default, `-x`)
- Octal (`-o`, `-b`)
- Decimal (`-d`)
- Character (`-c`, `-X`)
- Binary (`-a`)
- Colored output (`-L`)

### Display Options

- ASCII representation (default, disable with `-i`)
- Line squeezing for identical lines (`-w`)
- Arbitrary skip/length with size suffixes (`KiB, MiB, GiB, TiB, PiB, EiB, ZiB, YiB`)

## Example Output

```
❯ fasthex /bin/ls | head
00000000 7f 45 4c 46 02 01 01 00  00 00 00 00 00 00 00 00  .ELF............
00000010 03 00 3e 00 01 00 00 00  f0 54 00 00 00 00 00 00  ..>......T......
00000020 40 00 00 00 00 00 00 00  a8 73 02 00 00 00 00 00  @........s......
00000030 00 00 00 00 40 00 38 00  0e 00 40 00 1c 00 1b 00  ....@.8...@.....
00000040 06 00 00 00 04 00 00 00  40 00 00 00 00 00 00 00  ........@.......
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


### Output Format

Standard row layout – 76 bytes:

```
[0..7]   8-digit hex offset
[8]      ' '
[9..32]  hex group-1  (8 bytes × "XX ")
[33]     extra gap space
[34..57] hex group-2  (8 bytes × "XX ")
[58]     ' '
[59..74] ASCII printable / '.'
[75]     '\n'
```

### Full Help

```
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
```

## Known Limitations

- Offset field is u32: Files larger than 4 GiB will have wrapping offsets (e.g., a 5 GiB file offsets cycle from `0xFFFFFFFF` back to `0x00000000`)

## Testing Conditions

- Hardware: Intel i5-7500T, 16GB DDR4 RAM, Samsung 990 Pro NVMe SSD
- OS Settings: `iommu.passthrough=0`, `iommu.strict=1` (you may get better results on the same hardware)
- Output: All tests redirected to `/dev/null`
- Flags: No additional unmentioned flags were used
- I use Arch GNU/Linux btw
