//! fasthex – a very fast hex dumper
//!
//! Speed advantages:
//!   1. mmap path: output formatted in parallel with rayon in 64 MiB chunks.
//!   2. AVX2 path: processes 32 bytes (2 rows) per SIMD call; falls back to
//!      SSE4.1/SSSE3 (16 bytes / 1 row) or scalar.
//!   3. Double-buffered I/O: a dedicated writer thread drains completed chunks
//!      while rayon formats the next one.
//!   4. MADV_SEQUENTIAL on open + MADV_WILLNEED two chunks ahead to hide
//!      mmap page-fault latency.
//!   5. Zero-copy output via an internal pipe pair:
//!        formatted buffer
//!          → vmsplice  (userspace pages → kernel pipe, zero-copy)
//!          → splice    (kernel pipe → stdout fd, zero-copy)
//!      This path works regardless of what stdout is (/dev/null, file, pipe,
//!      socket) because we own the intermediate pipe. Falls back to write_all
//!      if splice rejects the stdout fd (e.g. a tty).
//!   6. Streaming (stdin) path uses a 4 MiB write buffer.

#![allow(clippy::missing_safety_doc)]

use memmap2::Mmap;
use rayon::prelude::*;
use std::arch::x86_64::*;
use std::env;
use std::fs::File;
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::sync::mpsc::{channel, sync_channel};
use std::thread;

const READ_BUF: usize = 256 * 1024;
const WRITE_BUF: usize = 4 * 1024 * 1024;
const PIPE_SIZE_HINT: libc::c_int = 2 * 1024 * 1024;

static HEX: &[u8; 16] = b"0123456789abcdef";

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Canonical,
    OneByteOctal,
    OneByteHex,
    OneByteChar,
    TwoBytesDecimal,
    TwoBytesOctal,
    TwoBytesHex,
    Binary,
}

impl Mode {
    fn bytes_per_row(self) -> usize {
        match self {
            Mode::Binary => 8,
            _ => 16,
        }
    }

    fn output_row_bytes(self, no_ascii: bool) -> usize {
        match self {
            Mode::Canonical if no_ascii => 59,
            Mode::Canonical => 76,
            Mode::Binary => 81,
            _ => 73,
        }
    }
}

struct Options {
    mode: Mode,
    color: bool,
    color_val: Vec<u8>,
    length: Option<u64>,
    skip: u64,
    squeezing: bool,
    no_ascii: bool,
    file: Option<String>,
}

fn print_help() {
    print!(
        "\
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
 GiB, TiB, PiB, EiB, ZiB, or YiB (where the \"iB\" is optional).
"
    );
}

fn parse_size(s: &str) -> Result<u64, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("empty value".into());
    }
    let num_end = s
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(s.len());
    if num_end == 0 {
        return Err(format!("invalid number: {}", s));
    }
    let num: u64 = s[..num_end]
        .parse()
        .map_err(|_| format!("invalid number: {}", s))?;
    let suffix = &s[num_end..];
    let mul: u64 = match suffix.to_ascii_lowercase().as_str() {
        "" => 1,
        "k" | "kib" => 1024,
        "m" | "mib" => 1024 * 1024,
        "g" | "gib" => 1024 * 1024 * 1024,
        "t" | "tib" => 1u64 << 40,
        "p" | "pib" => 1u64 << 50,
        "e" | "eib" => 1u64 << 60,
        "z" | "zib" => return Err("value too large".into()),
        "y" | "yib" => return Err("value too large".into()),
        _ => return Err(format!("unknown suffix: {}", suffix)),
    };
    num.checked_mul(mul)
        .ok_or_else(|| "value too large".into())
}

fn parse_hex_color(hex: &str) -> Option<Vec<u8>> {
    if hex.len() != 6 {
        return None;
    }
    let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
    let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
    let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
    Some(format!("\x1b[38;2;{};{};{}m", r, g, b).into_bytes())
}

fn parse_args() -> Result<Options, String> {
    let args: Vec<String> = env::args().collect();
    let mut opts = Options {
        mode: Mode::Canonical,
        color: false,
        color_val: b"\x1b[32m".to_vec(),
        length: None,
        skip: 0,
        squeezing: false,
        no_ascii: false,
        file: None,
    };

    let mut i = 1;
    while i < args.len() {
        let arg = &args[i];

        if arg == "--" {
            i += 1;
            if i < args.len() {
                opts.file = Some(args[i].clone());
            }
            break;
        }

        if arg.starts_with("--") {
            let opt = &arg[2..];
            if opt == "help" {
                print_help();
                std::process::exit(0);
            } else if opt == "version" {
                println!("2.0");
                std::process::exit(0);
            } else if opt == "one-byte-octal" {
                opts.mode = Mode::OneByteOctal;
            } else if opt == "one-byte-hex" {
                opts.mode = Mode::OneByteHex;
            } else if opt == "one-byte-char" {
                opts.mode = Mode::OneByteChar;
            } else if opt == "two-bytes-decimal" {
                opts.mode = Mode::TwoBytesDecimal;
            } else if opt == "two-bytes-octal" {
                opts.mode = Mode::TwoBytesOctal;
            } else if opt == "two-bytes-hex" {
                opts.mode = Mode::TwoBytesHex;
            } else if opt == "color" {
                opts.color = true;
            } else if let Some(val) = opt.strip_prefix("color=") {
                opts.color = true;
                if let Some(c) = parse_hex_color(val) {
                    opts.color_val = c;
                }
            } else if opt == "length" {
                i += 1;
                let v = args.get(i).ok_or("missing value for --length")?;
                opts.length = Some(parse_size(v)?);
            } else if let Some(val) = opt.strip_prefix("length=") {
                opts.length = Some(parse_size(val)?);
            } else if opt == "skip" {
                i += 1;
                let v = args.get(i).ok_or("missing value for --skip")?;
                opts.skip = parse_size(v)?;
            } else if let Some(val) = opt.strip_prefix("skip=") {
                opts.skip = parse_size(val)?;
            } else if opt == "with-squeezing" {
                opts.squeezing = true;
            } else if opt == "no-ascii" {
                opts.no_ascii = true;
            } else if opt == "binary" {
                opts.mode = Mode::Binary;
            } else {
                return Err(format!("unknown option: --{}", opt));
            }
        } else if arg.starts_with('-') && arg.len() > 1 {
            let bytes = arg[1..].as_bytes();
            let mut j = 0;
            while j < bytes.len() {
                match bytes[j] {
                    b'h' => {
                        print_help();
                        std::process::exit(0);
                    }
                    b'v' => {
                        println!("2.0");
                        std::process::exit(0);
                    }
                    b'b' => opts.mode = Mode::OneByteOctal,
                    b'X' => opts.mode = Mode::OneByteHex,
                    b'c' => opts.mode = Mode::OneByteChar,
                    b'd' => opts.mode = Mode::TwoBytesDecimal,
                    b'o' => opts.mode = Mode::TwoBytesOctal,
                    b'x' => opts.mode = Mode::TwoBytesHex,
                    b'L' => {
                        opts.color = true;
                        if j + 1 < bytes.len() && bytes[j + 1] == b'=' {
                            let val_str = String::from_utf8_lossy(&bytes[j + 2..]).into_owned();
                            if let Some(c) = parse_hex_color(&val_str) {
                                opts.color_val = c;
                            }
                            break;
                        }
                    }
                    b'w' => opts.squeezing = true,
                    b'i' => opts.no_ascii = true,
                    b'a' => opts.mode = Mode::Binary,
                    b'n' | b's' => {
                        let is_skip = bytes[j] == b's';
                        let val_str = if j + 1 < bytes.len() {
                            String::from_utf8_lossy(&bytes[j + 1..]).into_owned()
                        } else {
                            i += 1;
                            args.get(i)
                                .cloned()
                                .ok_or(if is_skip {
                                    "missing value for -s"
                                } else {
                                    "missing value for -n"
                                })?
                        };
                        let v = parse_size(&val_str)?;
                        if is_skip {
                            opts.skip = v;
                        } else {
                            opts.length = Some(v);
                        }
                        break; // consumed rest of this arg
                    }
                    _ => return Err(format!("unknown option: -{}", bytes[j] as char)),
                }
                j += 1;
            }
        } else {
            opts.file = Some(arg.clone());
        }
        i += 1;
    }

    Ok(opts)
}

struct ZeroCopyWriter {
    pipe_r: libc::c_int,
    pipe_w: libc::c_int,
    stdout: libc::c_int,
    fallback: bool,
}

impl ZeroCopyWriter {
    fn new() -> io::Result<Self> {
        let mut fds = [0i32; 2];
        if unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
            return Err(io::Error::last_os_error());
        }
        let (pipe_r, pipe_w) = (fds[0], fds[1]);
        unsafe { libc::fcntl(pipe_w, libc::F_SETPIPE_SZ, PIPE_SIZE_HINT) };
        Ok(Self {
            pipe_r,
            pipe_w,
            stdout: libc::STDOUT_FILENO,
            fallback: false,
        })
    }

    fn write_chunk(&mut self, buf: &[u8]) -> io::Result<()> {
        if self.fallback {
            return self.write_fallback(buf);
        }
        unsafe { self.write_zero_copy(buf) }
    }

    unsafe fn write_zero_copy(&mut self, buf: &[u8]) -> io::Result<()> {
        let mut vsrc = buf.as_ptr();
        let mut vremain = buf.len();
        while vremain > 0 {
            let iov = libc::iovec {
                iov_base: vsrc as *mut libc::c_void,
                iov_len: vremain,
            };
            let vspliced = libc::vmsplice(self.pipe_w, &iov, 1, libc::SPLICE_F_GIFT);
            if vspliced < 0 {
                self.drain_pipe_to_fallback(buf)?;
                return Ok(());
            }
            let vspliced = vspliced as usize;
            vsrc = vsrc.add(vspliced);
            vremain -= vspliced;
            let mut sremain = vspliced;
            while sremain > 0 {
                let spliced = libc::splice(
                    self.pipe_r,
                    std::ptr::null_mut(),
                    self.stdout,
                    std::ptr::null_mut(),
                    sremain,
                    libc::SPLICE_F_MOVE,
                );
                if spliced < 0 {
                    let err = io::Error::last_os_error();
                    if err.raw_os_error() == Some(libc::EINVAL) {
                        self.drain_pipe_to_fallback(buf)?;
                        return Ok(());
                    }
                    return Err(err);
                }
                sremain -= spliced as usize;
            }
        }
        Ok(())
    }

    fn drain_pipe_to_fallback(&mut self, full_buf: &[u8]) -> io::Result<()> {
        self.fallback = true;
        let mut tmp = vec![0u8; 65536];
        loop {
            let n = unsafe {
                libc::read(
                    self.pipe_r,
                    tmp.as_mut_ptr() as *mut libc::c_void,
                    tmp.len(),
                )
            };
            if n <= 0 {
                break;
            }
            io::stdout().lock().write_all(&tmp[..n as usize])?;
        }
        self.write_fallback(full_buf)
    }

    fn write_fallback(&self, buf: &[u8]) -> io::Result<()> {
        io::stdout().lock().write_all(buf)
    }
}

impl Drop for ZeroCopyWriter {
    fn drop(&mut self) {
        unsafe {
            libc::close(self.pipe_r);
            libc::close(self.pipe_w);
        }
    }
}

#[inline(always)]
unsafe fn write_offset(dst: *mut u8, off: u32) {
    *dst.add(0) = HEX[((off >> 28) & 0xf) as usize];
    *dst.add(1) = HEX[((off >> 24) & 0xf) as usize];
    *dst.add(2) = HEX[((off >> 20) & 0xf) as usize];
    *dst.add(3) = HEX[((off >> 16) & 0xf) as usize];
    *dst.add(4) = HEX[((off >> 12) & 0xf) as usize];
    *dst.add(5) = HEX[((off >> 8) & 0xf) as usize];
    *dst.add(6) = HEX[((off >> 4) & 0xf) as usize];
    *dst.add(7) = HEX[(off & 0xf) as usize];
}

fn write_offset_buf(dst: &mut [u8], off: u32) {
    unsafe { write_offset(dst.as_mut_ptr(), off) };
}

macro_rules! expand_and_store {
    ($dst:expr, $pairs_lo:expr, $pairs_hi:expr, $ascii:expr) => {{
        let dst_base: *mut u8 = $dst;
        let spaces = _mm_set1_epi8(b' ' as i8);
        let zero = _mm_setzero_si128();
        let shuf_a = _mm_setr_epi8(0, 1, -1, 2, 3, -1, 4, 5, -1, 6, 7, -1, -1, -1, -1, -1);
        let shuf_b = _mm_setr_epi8(8, 9, -1, 10, 11, -1, 12, 13, -1, 14, 15, -1, -1, -1, -1, -1);
        macro_rules! expand {
            ($pairs:expr, $shuf:expr) => {{
                let c = _mm_shuffle_epi8($pairs, $shuf);
                _mm_blendv_epi8(c, spaces, _mm_cmpeq_epi8(c, zero))
            }};
        }
        let chunk1 = expand!($pairs_lo, shuf_a);
        let chunk2 = expand!($pairs_lo, shuf_b);
        let chunk3 = expand!($pairs_hi, shuf_a);
        let chunk4 = expand!($pairs_hi, shuf_b);
        let p = dst_base.add(9);
        _mm_storeu_si128(p.add(0) as *mut __m128i, chunk1);
        _mm_storeu_si128(p.add(12) as *mut __m128i, chunk2);
        *p.add(24) = b' ';
        _mm_storeu_si128(p.add(25) as *mut __m128i, chunk3);
        _mm_storeu_si128(p.add(37) as *mut __m128i, chunk4);
        *p.add(49) = b' ';
        _mm_storeu_si128(dst_base.add(59) as *mut __m128i, $ascii);
        *dst_base.add(75) = b'\n';
    }};
}

#[target_feature(enable = "avx2,ssse3,sse4.1")]
unsafe fn format_two_rows_avx2(dst: *mut u8, src: *const u8, off: u32) {
    write_offset(dst, off);
    *dst.add(8) = b' ';
    write_offset(dst.add(76), off.wrapping_add(16));
    *dst.add(76 + 8) = b' ';

    let input = _mm256_loadu_si256(src as *const __m256i);
    let lo_mask = _mm256_set1_epi8(0x0f_u8 as i8);
    let lut = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        b'0' as i8, b'1' as i8, b'2' as i8, b'3' as i8, b'4' as i8, b'5' as i8, b'6' as i8,
        b'7' as i8, b'8' as i8, b'9' as i8, b'a' as i8, b'b' as i8, b'c' as i8, b'd' as i8,
        b'e' as i8, b'f' as i8,
    ));

    let lo_nib = _mm256_and_si256(input, lo_mask);
    let hi_nib = _mm256_and_si256(_mm256_srli_epi16(input, 4), lo_mask);
    let hex_lo = _mm256_shuffle_epi8(lut, lo_nib);
    let hex_hi = _mm256_shuffle_epi8(lut, hi_nib);
    let pairs_lo = _mm256_unpacklo_epi8(hex_hi, hex_lo);
    let pairs_hi = _mm256_unpackhi_epi8(hex_hi, hex_lo);

    let r0_plo = _mm256_castsi256_si128(pairs_lo);
    let r0_phi = _mm256_castsi256_si128(pairs_hi);
    let r1_plo = _mm256_extracti128_si256(pairs_lo, 1);
    let r1_phi = _mm256_extracti128_si256(pairs_hi, 1);

    let dot = _mm256_set1_epi8(b'.' as i8);
    let low = _mm256_set1_epi8(0x1f_u8 as i8);
    let high = _mm256_set1_epi8(0x7f_u8 as i8);
    let printable = _mm256_and_si256(
        _mm256_cmpgt_epi8(input, low),
        _mm256_cmpgt_epi8(high, input),
    );
    let ascii = _mm256_blendv_epi8(dot, input, printable);
    let ascii_r0 = _mm256_castsi256_si128(ascii);
    let ascii_r1 = _mm256_extracti128_si256(ascii, 1);

    expand_and_store!(dst, r0_plo, r0_phi, ascii_r0);
    expand_and_store!(dst.add(76), r1_plo, r1_phi, ascii_r1);
}

#[target_feature(enable = "ssse3,sse4.1")]
unsafe fn format_row_simd(dst: *mut u8, src: *const u8, off: u32) {
    write_offset(dst, off);
    *dst.add(8) = b' ';

    let input = _mm_loadu_si128(src as *const __m128i);
    let lo_mask = _mm_set1_epi8(0x0f_u8 as i8);
    let lut = _mm_setr_epi8(
        b'0' as i8, b'1' as i8, b'2' as i8, b'3' as i8, b'4' as i8, b'5' as i8, b'6' as i8,
        b'7' as i8, b'8' as i8, b'9' as i8, b'a' as i8, b'b' as i8, b'c' as i8, b'd' as i8,
        b'e' as i8, b'f' as i8,
    );

    let lo_nib = _mm_and_si128(input, lo_mask);
    let hi_nib = _mm_and_si128(_mm_srli_epi16(input, 4), lo_mask);
    let hex_lo = _mm_shuffle_epi8(lut, lo_nib);
    let hex_hi = _mm_shuffle_epi8(lut, hi_nib);
    let pairs_lo = _mm_unpacklo_epi8(hex_hi, hex_lo);
    let pairs_hi = _mm_unpackhi_epi8(hex_hi, hex_lo);

    let printable = _mm_and_si128(
        _mm_cmpgt_epi8(input, _mm_set1_epi8(0x1f_u8 as i8)),
        _mm_cmpgt_epi8(_mm_set1_epi8(0x7f_u8 as i8), input),
    );
    let ascii = _mm_blendv_epi8(_mm_set1_epi8(b'.' as i8), input, printable);

    expand_and_store!(dst, pairs_lo, pairs_hi, ascii);
}

fn format_row_canonical(dst: &mut [u8], src: &[u8], off: u32) {
    let n = src.len();
    write_offset_buf(dst, off);
    dst[8] = b' ';
    dst[9..59].fill(b' ');
    for i in 0..n {
        let pos = if i < 8 { i * 3 } else { 25 + (i - 8) * 3 };
        dst[9 + pos] = HEX[(src[i] >> 4) as usize];
        dst[9 + pos + 1] = HEX[(src[i] & 0xf) as usize];
    }
    for i in 0..16 {
        dst[59 + i] = if i < n {
            let c = src[i];
            if c >= 0x20 && c <= 0x7e {
                c
            } else {
                b'.'
            }
        } else {
            b' '
        };
    }
    dst[75] = b'\n';
}

fn format_row_canonical_no_ascii(dst: &mut [u8], src: &[u8], off: u32) {
    let n = src.len();
    write_offset_buf(dst, off);
    dst[8] = b' ';
    dst[9..58].fill(b' ');
    for i in 0..n {
        let pos = if i < 8 { i * 3 } else { 25 + (i - 8) * 3 };
        dst[9 + pos] = HEX[(src[i] >> 4) as usize];
        dst[9 + pos + 1] = HEX[(src[i] & 0xf) as usize];
    }
    dst[58] = b'\n';
}

fn format_row_one_byte_octal(dst: &mut [u8], src: &[u8], off: u32) {
    let n = src.len();
    write_offset_buf(dst, off);
    let mut pos = 8;
    for i in 0..16 {
        if i < n {
            let b = src[i];
            dst[pos] = b' ';
            dst[pos + 1] = b'0' + (b >> 6);
            dst[pos + 2] = b'0' + ((b >> 3) & 7);
            dst[pos + 3] = b'0' + (b & 7);
        } else {
            dst[pos] = b' ';
            dst[pos + 1] = b' ';
            dst[pos + 2] = b' ';
            dst[pos + 3] = b' ';
        }
        pos += 4;
    }
    dst[pos] = b'\n';
}

fn format_row_one_byte_hex(dst: &mut [u8], src: &[u8], off: u32) {
    let n = src.len();
    write_offset_buf(dst, off);
    let mut pos = 8;
    for i in 0..16 {
        if i < n {
            dst[pos] = b' ';
            dst[pos + 1] = b' ';
            dst[pos + 2] = HEX[(src[i] >> 4) as usize];
            dst[pos + 3] = HEX[(src[i] & 0xf) as usize];
        } else {
            dst[pos] = b' ';
            dst[pos + 1] = b' ';
            dst[pos + 2] = b' ';
            dst[pos + 3] = b' ';
        }
        pos += 4;
    }
    dst[pos] = b'\n';
}

fn format_row_one_byte_char(dst: &mut [u8], src: &[u8], off: u32) {
    let n = src.len();
    write_offset_buf(dst, off);
    let mut pos = 8;
    for i in 0..16 {
        if i < n {
            let b = src[i];
            match b {
                0x00 => dst[pos..pos + 4].copy_from_slice(b"  \\0"),
                0x07 => dst[pos..pos + 4].copy_from_slice(b"  \\a"),
                0x08 => dst[pos..pos + 4].copy_from_slice(b"  \\b"),
                0x09 => dst[pos..pos + 4].copy_from_slice(b"  \\t"),
                0x0a => dst[pos..pos + 4].copy_from_slice(b"  \\n"),
                0x0b => dst[pos..pos + 4].copy_from_slice(b"  \\v"),
                0x0c => dst[pos..pos + 4].copy_from_slice(b"  \\f"),
                0x0d => dst[pos..pos + 4].copy_from_slice(b"  \\r"),
                0x20..=0x7e => {
                    dst[pos] = b' ';
                    dst[pos + 1] = b' ';
                    dst[pos + 2] = b' ';
                    dst[pos + 3] = b;
                }
                _ => {
                    dst[pos] = b' ';
                    dst[pos + 1] = b'0' + (b >> 6);
                    dst[pos + 2] = b'0' + ((b >> 3) & 7);
                    dst[pos + 3] = b'0' + (b & 7);
                }
            }
        } else {
            dst[pos] = b' ';
            dst[pos + 1] = b' ';
            dst[pos + 2] = b' ';
            dst[pos + 3] = b' ';
        }
        pos += 4;
    }
    dst[pos] = b'\n';
}

#[inline(always)]
fn read_u16_le(src: &[u8], idx: usize) -> u16 {
    if idx + 1 < src.len() {
        u16::from_le_bytes([src[idx], src[idx + 1]])
    } else {
        src[idx] as u16
    }
}

fn write_u16_decimal(dst: &mut [u8], val: u16) {
    dst[0] = b' ';
    dst[1] = b' ';
    dst[2] = b' ';
    let mut v = val;
    dst[7] = b'0' + (v % 10) as u8;
    v /= 10;
    dst[6] = b'0' + (v % 10) as u8;
    v /= 10;
    dst[5] = b'0' + (v % 10) as u8;
    v /= 10;
    dst[4] = b'0' + (v % 10) as u8;
    v /= 10;
    dst[3] = b'0' + (v % 10) as u8;
}

fn format_row_two_bytes_decimal(dst: &mut [u8], src: &[u8], off: u32) {
    let n = src.len();
    write_offset_buf(dst, off);
    let mut pos = 8;
    for i in 0..8 {
        let bi = i * 2;
        if bi < n {
            write_u16_decimal(&mut dst[pos..pos + 8], read_u16_le(src, bi));
        } else {
            dst[pos..pos + 8].fill(b' ');
        }
        pos += 8;
    }
    dst[pos] = b'\n';
}

fn format_row_two_bytes_octal(dst: &mut [u8], src: &[u8], off: u32) {
    let n = src.len();
    write_offset_buf(dst, off);
    let mut pos = 8;
    for i in 0..8 {
        let bi = i * 2;
        if bi < n {
            let v = read_u16_le(src, bi) as u32;
            dst[pos] = b' ';
            dst[pos + 1] = b' ';
            dst[pos + 2] = b'0' + ((v >> 15) & 7) as u8;
            dst[pos + 3] = b'0' + ((v >> 12) & 7) as u8;
            dst[pos + 4] = b'0' + ((v >> 9) & 7) as u8;
            dst[pos + 5] = b'0' + ((v >> 6) & 7) as u8;
            dst[pos + 6] = b'0' + ((v >> 3) & 7) as u8;
            dst[pos + 7] = b'0' + (v & 7) as u8;
        } else {
            dst[pos..pos + 8].fill(b' ');
        }
        pos += 8;
    }
    dst[pos] = b'\n';
}

fn format_row_two_bytes_hex(dst: &mut [u8], src: &[u8], off: u32) {
    let n = src.len();
    write_offset_buf(dst, off);
    let mut pos = 8;
    for i in 0..8 {
        let bi = i * 2;
        if bi < n {
            let v = read_u16_le(src, bi);
            dst[pos] = b' ';
            dst[pos + 1] = b' ';
            dst[pos + 2] = b' ';
            dst[pos + 3] = b' ';
            dst[pos + 4] = HEX[((v >> 12) & 0xf) as usize];
            dst[pos + 5] = HEX[((v >> 8) & 0xf) as usize];
            dst[pos + 6] = HEX[((v >> 4) & 0xf) as usize];
            dst[pos + 7] = HEX[(v & 0xf) as usize];
        } else {
            dst[pos..pos + 8].fill(b' ');
        }
        pos += 8;
    }
    dst[pos] = b'\n';
}

fn format_row_binary(dst: &mut [u8], src: &[u8], off: u32) {
    let n = src.len();
    write_offset_buf(dst, off);
    dst[8] = b' ';
    for i in 0..8 {
        let base = 9 + i * 9;
        dst[base] = b' ';
        if i < n {
            let b = src[i];
            dst[base + 1] = b'0' + ((b >> 7) & 1);
            dst[base + 2] = b'0' + ((b >> 6) & 1);
            dst[base + 3] = b'0' + ((b >> 5) & 1);
            dst[base + 4] = b'0' + ((b >> 4) & 1);
            dst[base + 5] = b'0' + ((b >> 3) & 1);
            dst[base + 6] = b'0' + ((b >> 2) & 1);
            dst[base + 7] = b'0' + ((b >> 1) & 1);
            dst[base + 8] = b'0' + (b & 1);
        } else {
            dst[base + 1..base + 9].fill(b' ');
        }
    }
    dst[80] = b'\n';
}

fn format_row_dispatch(mode: Mode, no_ascii: bool, dst: &mut [u8], src: &[u8], off: u32) {
    match mode {
        Mode::Canonical if no_ascii => format_row_canonical_no_ascii(dst, src, off),
        Mode::Canonical => format_row_canonical(dst, src, off),
        Mode::OneByteOctal => format_row_one_byte_octal(dst, src, off),
        Mode::OneByteHex => format_row_one_byte_hex(dst, src, off),
        Mode::OneByteChar => format_row_one_byte_char(dst, src, off),
        Mode::TwoBytesDecimal => format_row_two_bytes_decimal(dst, src, off),
        Mode::TwoBytesOctal => format_row_two_bytes_octal(dst, src, off),
        Mode::TwoBytesHex => format_row_two_bytes_hex(dst, src, off),
        Mode::Binary => format_row_binary(dst, src, off),
    }
}

fn write_offset_colored(out: &mut impl Write, off: u32) -> io::Result<()> {
    out.write_all(b"\x1b[36m")?;
    let buf = [
        HEX[((off >> 28) & 0xf) as usize],
        HEX[((off >> 24) & 0xf) as usize],
        HEX[((off >> 20) & 0xf) as usize],
        HEX[((off >> 16) & 0xf) as usize],
        HEX[((off >> 12) & 0xf) as usize],
        HEX[((off >> 8) & 0xf) as usize],
        HEX[((off >> 4) & 0xf) as usize],
        HEX[(off & 0xf) as usize],
    ];
    out.write_all(&buf)?;
    out.write_all(b"\x1b[0m")
}

fn byte_color(b: u8, color_val: &[u8]) -> &[u8] {
    if b == 0 {
        b"\x1b[90m"
    } else if (0x20..=0x7e).contains(&b) {
        color_val
    } else {
        b"\x1b[33m"
    }
}

fn write_row_colored_canonical(
    out: &mut impl Write,
    src: &[u8],
    off: u32,
    no_ascii: bool,
    color_val: &[u8],
) -> io::Result<()> {
    write_offset_colored(out, off)?;
    out.write_all(b" ")?;
    for i in 0..16 {
        if i == 8 {
            out.write_all(b" ")?;
        }
        if i < src.len() {
            let b = src[i];
            out.write_all(byte_color(b, color_val))?;
            out.write_all(&[HEX[(b >> 4) as usize], HEX[(b & 0xf) as usize]])?;
            out.write_all(b"\x1b[0m ")?;
        } else {
            out.write_all(b"   ")?;
        }
    }
    if !no_ascii {
        out.write_all(b" ")?;
        for i in 0..16 {
            if i < src.len() {
                let b = src[i];
                if (0x20..=0x7e).contains(&b) {
                    out.write_all(b"\x1b[32m")?;
                    out.write_all(&[b])?;
                    out.write_all(b"\x1b[0m")?;
                } else {
                    out.write_all(b"\x1b[90m.\x1b[0m")?;
                }
            } else {
                out.write_all(b" ")?;
            }
        }
    }
    out.write_all(b"\n")
}

fn write_row_colored_generic(
    out: &mut impl Write,
    formatted: &[u8],
    color_val: &[u8],
) -> io::Result<()> {
    out.write_all(b"\x1b[36m")?;
    out.write_all(&formatted[..8])?;
    out.write_all(b"\x1b[0m")?;
    out.write_all(color_val)?;
    let end = formatted.len() - 1; // before newline
    out.write_all(&formatted[8..end])?;
    out.write_all(b"\x1b[0m\n")
}

fn write_row_colored(
    out: &mut impl Write,
    src: &[u8],
    off: u32,
    opts: &Options,
    row_buf: &mut [u8],
) -> io::Result<()> {
    match opts.mode {
        Mode::Canonical => {
            write_row_colored_canonical(out, src, off, opts.no_ascii, &opts.color_val)
        }
        _ => {
            format_row_dispatch(opts.mode, opts.no_ascii, row_buf, src, off);
            write_row_colored_generic(out, row_buf, &opts.color_val)
        }
    }
}

fn main() -> io::Result<()> {
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }

    let opts = match parse_args() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("fasthex: {}", e);
            std::process::exit(1);
        }
    };

    let use_avx2 = is_x86_feature_detected!("avx2");
    let use_simd =
        use_avx2 || (is_x86_feature_detected!("ssse3") && is_x86_feature_detected!("sse4.1"));

    let bpr = opts.mode.bytes_per_row();
    let orb = opts.mode.output_row_bytes(opts.no_ascii);

    let file_opt: Option<File> = match &opts.file {
        Some(path) => Some(File::open(path).map_err(|e| {
            io::Error::new(e.kind(), format!("{}: {}", path, e))
        })?),
        None => None,
    };

    if let Some(ref file) = file_opt {
        if let Ok(mmap) = unsafe { Mmap::map(file) } {
            let skip = (opts.skip as usize).min(mmap.len());
            let mut data = &mmap[skip..];
            if let Some(len) = opts.length {
                data = &data[..(len as usize).min(data.len())];
            }

            if data.is_empty() {
                return Ok(());
            }

            #[cfg(unix)]
            unsafe {
                libc::madvise(
                    data.as_ptr() as *mut libc::c_void,
                    data.len(),
                    libc::MADV_SEQUENTIAL,
                );
            }

            let start_off = opts.skip as u32;

            if opts.color || opts.squeezing {
                return run_serial_mmap(&opts, data, start_off, bpr, orb);
            } else {
                return run_parallel_mmap(&opts, data, start_off, bpr, orb, use_avx2, use_simd);
            }
        }
    }

    run_streaming(&opts, file_opt, bpr, orb, use_simd)
}

fn run_parallel_mmap(
    opts: &Options,
    data: &[u8],
    start_off: u32,
    bpr: usize,
    orb: usize,
    use_avx2: bool,
    use_simd: bool,
) -> io::Result<()> {
    let file_size = data.len();
    let full_rows = file_size / bpr;
    let tail_len = file_size % bpr;
    let chunk_rows = (64 * 1024 * 1024) / orb;
    let buf_cap = chunk_rows * orb;

    let (send_data, recv_data) = sync_channel::<Vec<u8>>(1);
    let (send_free, recv_free) = channel::<Vec<u8>>();
    send_free.send(vec![0u8; buf_cap]).unwrap();
    send_free.send(vec![0u8; buf_cap]).unwrap();

    let writer = thread::spawn(move || -> io::Result<()> {
        let mut zc = ZeroCopyWriter::new()?;
        while let Ok(chunk) = recv_data.recv() {
            zc.write_chunk(&chunk)?;
            let _ = send_free.send(chunk);
        }
        Ok(())
    });

    let mode = opts.mode;
    let no_ascii = opts.no_ascii;
    let can_use_canonical_simd = mode == Mode::Canonical && !no_ascii;

    let mut row_cursor = 0usize;
    while row_cursor < full_rows {
        let rows = (full_rows - row_cursor).min(chunk_rows);

        // Prefetch
        #[cfg(unix)]
        {
            let pf_start = (row_cursor + 2 * chunk_rows) * bpr;
            if pf_start < file_size {
                let pf_len = (chunk_rows * bpr).min(file_size - pf_start);
                unsafe {
                    libc::madvise(
                        data.as_ptr().add(pf_start) as *mut libc::c_void,
                        pf_len,
                        libc::MADV_WILLNEED,
                    );
                }
            }
        }

        let mut chunk_out = recv_free.recv().unwrap();
        chunk_out.resize(rows * orb, 0);

        if can_use_canonical_simd && use_avx2 {
            let even = rows & !1;
            chunk_out[..even * orb]
                .par_chunks_mut(orb * 2)
                .enumerate()
                .for_each(|(i, two_rows)| {
                    let src_off = (row_cursor + i * 2) * bpr;
                    let off = start_off.wrapping_add(src_off as u32);
                    unsafe {
                        format_two_rows_avx2(two_rows.as_mut_ptr(), data.as_ptr().add(src_off), off);
                    }
                });
            if rows & 1 != 0 {
                let src_off = (row_cursor + rows - 1) * bpr;
                let off = start_off.wrapping_add(src_off as u32);
                unsafe {
                    format_row_simd(
                        chunk_out[(rows - 1) * orb..].as_mut_ptr(),
                        data.as_ptr().add(src_off),
                        off,
                    );
                }
            }
        } else if can_use_canonical_simd && use_simd {
            chunk_out
                .par_chunks_mut(orb)
                .enumerate()
                .for_each(|(i, row)| {
                    let src_off = (row_cursor + i) * bpr;
                    let off = start_off.wrapping_add(src_off as u32);
                    unsafe {
                        format_row_simd(row.as_mut_ptr(), data.as_ptr().add(src_off), off);
                    }
                });
        } else {
            chunk_out
                .par_chunks_mut(orb)
                .enumerate()
                .for_each(|(i, row)| {
                    let src_off = (row_cursor + i) * bpr;
                    let off = start_off.wrapping_add(src_off as u32);
                    format_row_dispatch(mode, no_ascii, row, &data[src_off..src_off + bpr], off);
                });
        }

        send_data.send(chunk_out).unwrap();
        row_cursor += rows;
    }

    drop(send_data);
    writer.join().unwrap()?;

    if tail_len > 0 {
        let src_off = full_rows * bpr;
        let off = start_off.wrapping_add(src_off as u32);
        let mut row = vec![0u8; orb];
        format_row_dispatch(mode, no_ascii, &mut row, &data[src_off..], off);
        io::stdout().lock().write_all(&row)?;
    }

    Ok(())
}

fn run_serial_mmap(
    opts: &Options,
    data: &[u8],
    start_off: u32,
    bpr: usize,
    orb: usize,
) -> io::Result<()> {
    let file_size = data.len();
    let full_rows = file_size / bpr;
    let tail_len = file_size % bpr;

    let stdout = io::stdout();
    let mut out = BufWriter::with_capacity(WRITE_BUF, stdout.lock());
    let mut row_buf = vec![0u8; orb];

    let mut prev_row: Vec<u8> = Vec::new();
    let mut squeezed = false;

    for r in 0..full_rows {
        let src_off = r * bpr;
        let row_data = &data[src_off..src_off + bpr];
        let off = start_off.wrapping_add(src_off as u32);

        if opts.squeezing && row_data == prev_row.as_slice() {
            if !squeezed {
                out.write_all(b"*\n")?;
                squeezed = true;
            }
            continue;
        }
        if opts.squeezing {
            squeezed = false;
            prev_row.clear();
            prev_row.extend_from_slice(row_data);
        }

        if opts.color {
            write_row_colored(&mut out, row_data, off, opts, &mut row_buf)?;
        } else {
            format_row_dispatch(opts.mode, opts.no_ascii, &mut row_buf, row_data, off);
            out.write_all(&row_buf)?;
        }
    }

    if tail_len > 0 {
        let src_off = full_rows * bpr;
        let off = start_off.wrapping_add(src_off as u32);
        if opts.color {
            write_row_colored(&mut out, &data[src_off..], off, opts, &mut row_buf)?;
        } else {
            format_row_dispatch(opts.mode, opts.no_ascii, &mut row_buf, &data[src_off..], off);
            out.write_all(&row_buf)?;
        }
    }

    if opts.squeezing {
        let final_off = start_off.wrapping_add(file_size as u32);
        if opts.color {
            write_offset_colored(&mut out, final_off)?;
            out.write_all(b"\n")?;
        } else {
            write_offset_buf(&mut row_buf, final_off);
            out.write_all(&row_buf[..8])?;
            out.write_all(b"\n")?;
        }
    }

    out.flush()
}

fn run_streaming(
    opts: &Options,
    file_opt: Option<File>,
    bpr: usize,
    orb: usize,
    use_simd: bool,
) -> io::Result<()> {
    let mut file_opt_owned = file_opt;

    // Handle skip for seekable files
    if opts.skip > 0 {
        if let Some(ref mut f) = file_opt_owned {
            f.seek(SeekFrom::Start(opts.skip))?;
        }
    }

    let mut reader: Box<dyn Read> = match file_opt_owned {
        Some(f) => Box::new(f),
        None => Box::new(io::stdin()),
    };

    // Handle skip for stdin (non-seekable)
    if opts.skip > 0 && opts.file.is_none() {
        let mut skip_buf = vec![0u8; 8192];
        let mut to_skip = opts.skip;
        while to_skip > 0 {
            let chunk = to_skip.min(skip_buf.len() as u64) as usize;
            let n = reader.read(&mut skip_buf[..chunk])?;
            if n == 0 {
                break;
            }
            to_skip -= n as u64;
        }
    }

    let stdout = io::stdout();
    let mut out = BufWriter::with_capacity(WRITE_BUF, stdout.lock());
    let mut rbuf = vec![0u8; READ_BUF];
    let mut wbuf = vec![0u8; WRITE_BUF + orb];
    let mut wpos = 0usize;
    let mut offset = opts.skip as u32;
    let remaining = opts.length;

    let can_simd_canonical =
        use_simd && opts.mode == Mode::Canonical && !opts.no_ascii && !opts.color;

    let mut prev_row: Vec<u8> = Vec::new();
    let mut squeezed = false;
    let mut total_read = 0u64;

    loop {
        let max_read = match remaining {
            Some(rem) => rbuf.len().min(rem.saturating_sub(total_read) as usize),
            None => rbuf.len(),
        };
        if max_read == 0 {
            break;
        }

        let n = reader.read(&mut rbuf[..max_read])?;
        if n == 0 {
            break;
        }
        total_read += n as u64;

        let full = n / bpr;
        let tail = n % bpr;

        for r in 0..full {
            let src = &rbuf[r * bpr..(r + 1) * bpr];

            // Squeezing
            if opts.squeezing && src == prev_row.as_slice() {
                if !squeezed {
                    if wpos > 0 {
                        out.write_all(&wbuf[..wpos])?;
                        wpos = 0;
                    }
                    out.write_all(b"*\n")?;
                    squeezed = true;
                }
                offset = offset.wrapping_add(bpr as u32);
                continue;
            }
            if opts.squeezing {
                squeezed = false;
                prev_row.clear();
                prev_row.extend_from_slice(src);
            }

            if opts.color {
                if wpos > 0 {
                    out.write_all(&wbuf[..wpos])?;
                    wpos = 0;
                }
                let mut row_buf = vec![0u8; orb];
                write_row_colored(&mut out, src, offset, opts, &mut row_buf)?;
            } else {
                if wpos > WRITE_BUF - orb {
                    out.write_all(&wbuf[..wpos])?;
                    wpos = 0;
                }
                if can_simd_canonical {
                    unsafe {
                        format_row_simd(wbuf[wpos..].as_mut_ptr(), src.as_ptr(), offset);
                    }
                } else {
                    format_row_dispatch(
                        opts.mode,
                        opts.no_ascii,
                        &mut wbuf[wpos..wpos + orb],
                        src,
                        offset,
                    );
                }
                wpos += orb;
            }
            offset = offset.wrapping_add(bpr as u32);
        }

        if tail > 0 {
            let base = full * bpr;
            let src = &rbuf[base..base + tail];

            if opts.color {
                if wpos > 0 {
                    out.write_all(&wbuf[..wpos])?;
                    wpos = 0;
                }
                let mut row_buf = vec![0u8; orb];
                write_row_colored(&mut out, src, offset, opts, &mut row_buf)?;
            } else {
                if wpos > WRITE_BUF - orb {
                    out.write_all(&wbuf[..wpos])?;
                    wpos = 0;
                }
                format_row_dispatch(
                    opts.mode,
                    opts.no_ascii,
                    &mut wbuf[wpos..wpos + orb],
                    src,
                    offset,
                );
                wpos += orb;
            }
            offset = offset.wrapping_add(tail as u32);
        }
    }

    if wpos > 0 {
        out.write_all(&wbuf[..wpos])?;
    }

    if opts.squeezing {
        let final_off = offset;
        if opts.color {
            write_offset_colored(&mut out, final_off)?;
            out.write_all(b"\n")?;
        } else {
            let mut off_buf = [0u8; 9];
            write_offset_buf(&mut off_buf, final_off);
            off_buf[8] = b'\n';
            out.write_all(&off_buf)?;
        }
    }

    out.flush()
}
