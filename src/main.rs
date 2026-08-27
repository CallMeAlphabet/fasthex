#![feature(portable_simd)]
//! Copyright 2026 CallMeAlphabet (ItzAlphabet)
//!
//! Licensed under the Apache License, Version 2.0 (the "License");
//! you may not use this file except in compliance with the License.
//! You may obtain a copy of the License at
//!
//!    http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in writing, software
//! distributed under the License is distributed on an "AS IS" BASIS,
//! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//! See the License for the specific language governing permissions and
//! limitations under the License.

//! fasthex – a very fast hex dumper
//!
//! Speed notes:
//!   1. mmap + rayon parallel formatting in 64 MiB chunks.
//!   2. AVX2: 32 bytes (2 rows) per SIMD call; SSE4.1/SSSE3 fallback (16 bytes).
//!      Both paths only engage for canonical mode, width=16, group=1, no-border,
//!      no-color, no-uppercase, offset-hex, big-endian — the common fast path.
//!   3. Double-buffered I/O: a dedicated writer thread drains while rayon formats.
//!   4. MADV_SEQUENTIAL + MADV_WILLNEED two chunks ahead.
//!   5. vmsplice → splice zero-copy output path; falls back to write_all for ttys.
//!   6. Streaming path uses a 4 MiB write buffer.
//!   7. u64 offsets: 8 hex digits normally, grows naturally past 0xFFFFFFFF.
//!   8. FASTHEX_DEFAULT_OPTS env var prepended before argv.

#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use clihelp::{HelpPage, Row, Section};
use memmap2::Mmap;
use rayon::prelude::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{self, BufWriter, IsTerminal, Read, Seek, SeekFrom, Write};
use std::sync::mpsc::{channel, sync_channel};
use std::thread;

const READ_BUF: usize = 4 * 1024 * 1024;
const WRITE_BUF: usize = 4 * 1024 * 1024;
const PIPE_SIZE_HINT: libc::c_int = 2 * 1024 * 1024;
const _CHUNK_ROWS: usize = (64 * 1024 * 1024) / 76; // recalculated per-mode at runtime

static HEX_LOWER: &[u8; 16] = b"0123456789abcdef";
static HEX_UPPER: &[u8; 16] = b"0123456789ABCDEF";

use std::simd::Simd;

/// Portable nibble LUT hex encode of 16 bytes → 32 ASCII digits.
/// LLVM lowers this to AVX2/SSSE3 on x86_64 and NEON `tbl` on aarch64.
#[inline(always)]
fn encode_hex16_portable(src: &[u8], lut: &[u8; 16], dst: &mut [u8]) {
    type V = Simd<u8, 16>;
    let v = V::from_slice(&src[..16]);
    let lo = v & V::splat(0x0f);
    let hi = v >> 4;
    let lutv = V::from_array(*lut);
    let hlo = lutv.swizzle_dyn(lo);
    let hhi = lutv.swizzle_dyn(hi);
    let mut out = [0u8; 32];
    for i in 0..16 {
        out[i * 2] = hhi[i];
        out[i * 2 + 1] = hlo[i];
    }
    dst[..32].copy_from_slice(&out);
}

#[inline]
fn cpu_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}
#[inline]
fn cpu_sse41() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("ssse3") && is_x86_feature_detected!("sse4.1")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}
#[inline]
fn cpu_avx512() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

// CP437 table (256 entries, index = byte value)
static CP437: [char; 256] = [
    ' ','☺','☻','♥','♦','♣','♠','•','◘','○','◙','♂','♀','♪','♫','☼',
    '►','◄','↕','‼','¶','§','▬','↨','↑','↓','→','←','∟','↔','▲','▼',
    ' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/',
    '0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?',
    '@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O',
    'P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_',
    '`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o',
    'p','q','r','s','t','u','v','w','x','y','z','{','|','}','~','⌂',
    'Ç','ü','é','â','ä','à','å','ç','ê','ë','è','ï','î','ì','Ä','Å',
    'É','æ','Æ','ô','ö','ò','û','ù','ÿ','Ö','Ü','¢','£','¥','₧','ƒ',
    'á','í','ó','ú','ñ','Ñ','ª','º','¿','⌐','¬','½','¼','¡','«','»',
    '░','▒','▓','│','┤','╡','╢','╖','╕','╣','║','╗','╝','╜','╛','┐',
    '└','┴','┬','├','─','┼','╞','╟','╚','╔','╩','╦','╠','═','╬','╧',
    '╨','╤','╥','╙','╘','╒','╓','╫','╪','┘','┌','█','▄','▌','▐','▀',
    'α','ß','Γ','π','Σ','σ','µ','τ','Φ','Θ','Ω','δ','∞','φ','ε','∩',
    '≡','±','≥','≤','⌠','⌡','÷','≈','°','∙','·','√','ⁿ','²','■',' ',
];

// EBCDIC→ASCII table
static EBCDIC_TO_ASCII: [u8; 256] = {
    let mut t = [b'.'; 256];
    // printable EBCDIC ranges → ASCII
    let pairs: &[(u8,u8)] = &[
        (0x40,b' '),(0x4b,b'.'),(0x4c,b'<'),(0x4d,b'('),(0x4e,b'+'),(0x4f,b'|'),
        (0x50,b'&'),(0x5a,b'!'),(0x5b,b'$'),(0x5c,b'*'),(0x5d,b')'),(0x5e,b';'),
        (0x5f,b'^'),(0x60,b'-'),(0x61,b'/'),(0x6b,b','),(0x6c,b'%'),(0x6d,b'_'),
        (0x6e,b'>'),(0x6f,b'?'),(0x79,b'`'),(0x7a,b':'),(0x7b,b'#'),(0x7c,b'@'),
        (0x7d,b'\''),(0x7e,b'='),(0x7f,b'"'),
    ];
    let mut i = 0u8;
    while i < 10 { t[(0xf0 + i) as usize] = b'0' + i; i += 1; }
    i = 0;
    while i < 9  { t[(0xc1+i) as usize] = b'A'+i; i += 1; }
    i = 0;
    while i < 9  { t[(0xd1+i) as usize] = b'J'+i; i += 1; }
    i = 0;
    while i < 8  { t[(0xe2+i) as usize] = b'S'+i; i += 1; }
    i = 0;
    while i < 9  { t[(0x81+i) as usize] = b'a'+i; i += 1; }
    i = 0;
    while i < 9  { t[(0x91+i) as usize] = b'j'+i; i += 1; }
    i = 0;
    while i < 8  { t[(0xa2+i) as usize] = b's'+i; i += 1; }
    let mut p = 0;
    while p < pairs.len() { t[pairs[p].0 as usize] = pairs[p].1; p += 1; }
    t
};

#[derive(Clone, Copy, PartialEq, Debug)]
enum DisplayMode {
    Canonical,
    OneByteHex,        // -x
    TwoByteHex,        // -X
    OneByteOctal,      // -o
    TwoByteOctal,      // -O
    OneByteDecimal,    // -d
    TwoByteDecimal,    // -D
    OneByteChar,       // -c
    Binary,            // -b
    Plain,             // -p
    CInclude,          // -i
    Reverse,           // -r  (not a display mode per se, handled separately)
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum BorderStyle { None, Ascii, Unicode }

#[derive(Clone, Copy, PartialEq, Debug)]
enum ColorWhen { Auto, Always, Never }

#[derive(Clone, Copy, PartialEq, Debug)]
enum ColorScheme { Default, Type, Gradient }

#[derive(Clone, Copy, PartialEq, Debug)]
enum CharTable { Ascii, Default, Braille, Cp437, Ebcdic }

#[derive(Clone, Copy, PartialEq, Debug)]
enum Endian { Big, Little }

#[derive(Clone)]
struct Options {
    mode:         DisplayMode,
    width:        usize,        // bytes per row (0 = auto for unicode border)
    group:        usize,        // bytes per group: 1,2,4,8
    endian:       Endian,
    border:       BorderStyle,
    no_ascii:     bool,
    minimal:      bool,
    no_position:  bool,
    skip:         i64,          // signed: negative = from end
    length:       Option<u64>,
    jump:         i64,          // signed offset bias
    uppercase:    bool,
    offset_dec:   bool,
    color:        ColorWhen,
    scheme:       ColorScheme,
    table:        CharTable,
    squeeze:      bool,
    max_lines:    Option<u64>,
    quiet:        bool,
    // -F custom format strings
    formats:      Vec<String>,
    files:        Vec<String>,  // "-" means stdin
    // reverse-mode jump target
    reverse_jump: Option<i64>,
    // C include variable name (derived from first filename)
    include_name: Option<String>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            mode:         DisplayMode::Canonical,
            width:        16,
            group:        1,
            endian:       Endian::Big,
            border:       BorderStyle::None,
            no_ascii:     false,
            minimal:      false,
            no_position:  false,
            skip:         0,
            length:       None,
            jump:         0,
            uppercase:    false,
            offset_dec:   false,
            color:        ColorWhen::Auto,
            scheme:       ColorScheme::Default,
            table:        CharTable::Ascii,
            squeeze:      false,
            max_lines:    None,
            quiet:        false,
            formats:      Vec::new(),
            files:        Vec::new(),
            reverse_jump: None,
            include_name: None,
        }
    }
}

fn row(short: &'static str, long: &'static str, desc: &'static str) -> Row {
    Row::new(short, long, desc)
}
fn row_val(
    short: &'static str,
    long: &'static str,
    placeholder: &'static str,
    desc: &'static str,
) -> Row {
    Row::with_value(short, long, placeholder, desc)
}

fn output_format_rows() -> Vec<Row> {
    vec![
        row("", "(default)", "canonical hex + ASCII display"),
        row("-x", "--hex", "one-byte hexadecimal display"),
        row("-X", "--hex-wide", "two-byte hexadecimal display"),
        row("-o", "--octal", "one-byte octal display"),
        row("-O", "--octal-wide", "two-byte octal display"),
        row("-d", "--decimal", "one-byte decimal display"),
        row("-D", "--decimal-wide", "two-byte decimal display"),
        row("-c", "--chars", "one-byte character display"),
        row("-b", "--binary", "binary display (8 bits per byte)"),
        row("-p", "--plain", "plain hex string, no offset or ASCII"),
        row("-i", "--include", "C include file style output"),
        row("-r", "--reverse", "convert hex dump back to binary"),
    ]
}

fn layout_rows() -> Vec<Row> {
    vec![
        row_val("-W", "--width", "<N>", "bytes per row (default: 16)"),
        row_val("-g", "--group", "<N>", "bytes per group: 1, 2, 4, 8"),
        row_val("-E", "--endian", "<MODE>", "big | little  (default: big)"),
        row_val("-B", "--border", "<STYLE>", "none | ascii | unicode  (default: none)"),
        row("-A", "--no-ascii", "hide the ASCII panel"),
        row("-P", "--no-position", "hide the offset/position column"),
        row("", "--minimal", "compact rows: offset + hex + ascii, no separators"),
    ]
}

fn offset_nav_rows() -> Vec<Row> {
    vec![
        row_val("-s", "--skip", "<N>", "skip first N bytes (negative = from end)"),
        row_val("-n", "--length", "<N>", "read only N bytes"),
        row_val("-j", "--jump", "<N>", "bias added to every displayed offset"),
        row("-u", "--uppercase", "uppercase hex digits (A-F)"),
        row("", "--offset-dec", "show offsets in decimal"),
    ]
}

fn color_rows() -> Vec<Row> {
    vec![
        row_val("-L", "--color", "<WHEN>", "auto | always | never  (default: auto)"),
        row_val("-S", "--scheme", "<NAME>", "default | type | gradient"),
        row_val("-T", "--table", "<MODE>", "ascii | default | braille | cp437 | ebcdic"),
    ]
}

fn filtering_rows() -> Vec<Row> {
    vec![
        row("-w", "--squeeze", "replace identical rows with '*'"),
        row_val("-m", "--max-lines", "<N>", "stop after N output lines"),
        row("-q", "--quiet", "suppress warnings"),
    ]
}

fn custom_format_rows() -> Vec<Row> {
    vec![
        row_val("-F", "--format", "<FMT>", "hexdump -e style format string"),
        row_val("-f", "--format-file", "<FILE>", "read format strings from file"),
    ]
}

fn misc_rows() -> Vec<Row> {
    vec![
        row("-h", "--help", "show this help"),
        row("-v", "--version", "show version"),
    ]
}

fn sections() -> Vec<Section> {
    vec![
        Section {
            title: "OUTPUT FORMAT",
            note: Some("Rule: lowercase = one-byte mode, UPPERCASE = two-byte mode."),
            rows: output_format_rows(),
        },
        Section { title: "LAYOUT", note: None, rows: layout_rows() },
        Section { title: "OFFSET & NAVIGATION", note: None, rows: offset_nav_rows() },
        Section { title: "COLOR", note: None, rows: color_rows() },
        Section { title: "FILTERING & FLOW", note: None, rows: filtering_rows() },
        Section { title: "CUSTOM FORMAT", note: None, rows: custom_format_rows() },
        Section { title: "MISC", note: None, rows: misc_rows() },
    ]
}

fn print_help() {
    print_help_body(io::stdout().is_terminal());
}

pub fn print_help_body(on: bool) {
    let mut page = HelpPage::new(format!(
        "{} {} - {}",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
        env!("CARGO_PKG_DESCRIPTION")
    ))
        .usage("fasthex [options] [file]...")
        .usage("fasthex -r [options] [file] [-j <offset>]")
        .usage("fasthex [options] -          read from stdin explicitly")
        .blurb(
            "Multiple files are concatenated and treated as one stream.\n\
             If no file is given, reads from stdin.",
        )
        .footer("SIZE SUFFIXES: KiB/K/MiB/M/GiB/G/TiB/T/PiB/P/EiB/E  kB/MB/GB/TB/PB/EB  0x…");

    for section in sections() {
        page = page.section(section);
    }

    print!("{}", page.render(on));
}

fn parse_size_signed(s: &str) -> Result<i64, String> {
    let s = s.trim();
    if s.is_empty() { return Err("empty value".into()); }
    let negative = s.starts_with('-');
    let s2 = if negative { &s[1..] } else { s };
    let abs = parse_size_unsigned(s2)?;
    if abs > i64::MAX as u64 { return Err("value too large".into()); }
    Ok(if negative { -(abs as i64) } else { abs as i64 })
}

fn parse_size_unsigned(s: &str) -> Result<u64, String> {
    let s = s.trim();
    if s.is_empty() { return Err("empty value".into()); }

    if s.starts_with("0x") || s.starts_with("0X") {
        return u64::from_str_radix(&s[2..], 16)
            .map_err(|_| format!("invalid hex value: {}", s));
    }

    let num_end = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    if num_end == 0 { return Err(format!("invalid number: {}", s)); }
    let num: u64 = s[..num_end].parse()
        .map_err(|_| format!("invalid number: {}", s))?;
    let suffix = &s[num_end..];
    let mul: u64 = match suffix.to_lowercase().as_str() {
        "" => 1, "k" | "kib" => 1 << 10,
        "m" | "mib" => 1 << 20,
        "g" | "gib" => 1 << 30,
        "t" | "tib" => 1u64 << 40,
        "p" | "pib" => 1u64 << 50,
        "e" | "eib" => 1u64 << 60,
        "kb" => 1_000,
        "mb" => 1_000_000,
        "gb" => 1_000_000_000,
        "tb" => 1_000_000_000_000,
        "pb" => 1_000_000_000_000_000,
        "eb" => 1_000_000_000_000_000_000,
        _ => return Err(format!("unknown suffix: {}", suffix)),
    };
    num.checked_mul(mul).ok_or_else(|| "value too large".into())
}

fn parse_args_from(raw: &[String]) -> Result<Options, String> {
    let mut opts = Options::default();
    let mut i = 0usize;

    while i < raw.len() {
        let arg = &raw[i];

        if arg == "--" {
            i += 1;
            while i < raw.len() { opts.files.push(raw[i].clone()); i += 1; }
            break;
        }

        if arg.starts_with("--") {
            let key_val = &arg[2..];
            let (key, val_opt) = if let Some(eq) = key_val.find('=') {
                (&key_val[..eq], Some(&key_val[eq+1..]))
            } else {
                (key_val, None)
            };

            match key {
                "help"    => { print_help(); std::process::exit(0); }
                "version" => { println!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION")); std::process::exit(0); }
                "hex"           => opts.mode = DisplayMode::OneByteHex,
                "hex-wide"      => opts.mode = DisplayMode::TwoByteHex,
                "octal"         => opts.mode = DisplayMode::OneByteOctal,
                "octal-wide"    => opts.mode = DisplayMode::TwoByteOctal,
                "decimal"       => opts.mode = DisplayMode::OneByteDecimal,
                "decimal-wide"  => opts.mode = DisplayMode::TwoByteDecimal,
                "chars"         => opts.mode = DisplayMode::OneByteChar,
                "binary"        => opts.mode = DisplayMode::Binary,
                "plain"         => opts.mode = DisplayMode::Plain,
                "include"       => opts.mode = DisplayMode::CInclude,
                "reverse"       => opts.mode = DisplayMode::Reverse,
                "no-ascii"      => opts.no_ascii = true,
                "minimal"       => opts.minimal = true,
                "no-position"   => opts.no_position = true,
                "uppercase"     => opts.uppercase = true,
                "offset-dec"    => opts.offset_dec = true,
                "squeeze"       => opts.squeeze = true,
                "quiet"         => opts.quiet = true,
                "width" => {
                    let v = val_opt.unwrap_or_else(|| { i += 1; raw.get(i).map(|s| s.as_str()).unwrap_or("") });
                    opts.width = parse_size_unsigned(v)? as usize;
                    if opts.width == 0 { return Err("--width must be > 0".into()); }
                }
                "group" => {
                    let v = val_opt.unwrap_or_else(|| { i += 1; raw.get(i).map(|s| s.as_str()).unwrap_or("") });
                    opts.group = parse_size_unsigned(v)? as usize;
                    if !matches!(opts.group, 1|2|4|8) {
                        return Err("--group must be 1, 2, 4, or 8".into());
                    }
                }
                "endian" => {
                    let v = val_opt.unwrap_or_else(|| { i += 1; raw.get(i).map(|s| s.as_str()).unwrap_or("") });
                    opts.endian = match v {
                        "big"    => Endian::Big,
                        "little" => Endian::Little,
                        _ => return Err(format!("unknown endian: {}", v)),
                    };
                }
                "border" => {
                    let v = val_opt.unwrap_or_else(|| { i += 1; raw.get(i).map(|s| s.as_str()).unwrap_or("") });
                    opts.border = match v {
                        "none"    => BorderStyle::None,
                        "ascii"   => BorderStyle::Ascii,
                        "unicode" => BorderStyle::Unicode,
                        _ => return Err(format!("unknown border style: {}", v)),
                    };
                }
                "skip" => {
                    let v = val_opt.unwrap_or_else(|| { i += 1; raw.get(i).map(|s| s.as_str()).unwrap_or("") });
                    opts.skip = parse_size_signed(v)?;
                }
                "length" => {
                    let v = val_opt.unwrap_or_else(|| { i += 1; raw.get(i).map(|s| s.as_str()).unwrap_or("") });
                    opts.length = Some(parse_size_unsigned(v)?);
                }
                "jump" => {
                    let v = val_opt.unwrap_or_else(|| { i += 1; raw.get(i).map(|s| s.as_str()).unwrap_or("") });
                    opts.jump = parse_size_signed(v)?;
                }
                "color" => {
                    let v = val_opt.unwrap_or_else(|| { i += 1; raw.get(i).map(|s| s.as_str()).unwrap_or("auto") });
                    opts.color = parse_color_when(v)?;
                }
                "scheme" => {
                    let v = val_opt.unwrap_or_else(|| { i += 1; raw.get(i).map(|s| s.as_str()).unwrap_or("") });
                    opts.scheme = parse_scheme(v)?;
                }
                "table" => {
                    let v = val_opt.unwrap_or_else(|| { i += 1; raw.get(i).map(|s| s.as_str()).unwrap_or("") });
                    opts.table = parse_char_table(v)?;
                }
                "max-lines" => {
                    let v = val_opt.unwrap_or_else(|| { i += 1; raw.get(i).map(|s| s.as_str()).unwrap_or("") });
                    opts.max_lines = Some(parse_size_unsigned(v)?);
                }
                "format" => {
                    let v = val_opt.unwrap_or_else(|| { i += 1; raw.get(i).map(|s| s.as_str()).unwrap_or("") });
                    if !v.contains('"') {
                        return Err(format!("--format expects a quoted format string: {}", v));
                    }
                    opts.formats.push(v.to_string());
                }
                "format-file" => {
                    let v = val_opt.unwrap_or_else(|| { i += 1; raw.get(i).map(|s| s.as_str()).unwrap_or("") });
                    let content = std::fs::read_to_string(v)
                        .map_err(|e| format!("cannot read format file {}: {}", v, e))?;
                    for line in content.lines() {
                        let l = line.trim();
                        if !l.is_empty() { opts.formats.push(l.to_string()); }
                    }
                }
                _ => return Err(format!("unknown option: --{}", key)),
            }
        } else if arg.starts_with('-') && arg.len() > 1 {
            let bytes = arg[1..].as_bytes();
            let mut j = 0usize;
            while j < bytes.len() {
                match bytes[j] {
                    b'h' => { print_help(); std::process::exit(0); }
                    b'v' => { println!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION")); std::process::exit(0); }
                    b'x' => opts.mode = DisplayMode::OneByteHex,
                    b'X' => opts.mode = DisplayMode::TwoByteHex,
                    b'o' => opts.mode = DisplayMode::OneByteOctal,
                    b'O' => opts.mode = DisplayMode::TwoByteOctal,
                    b'd' => opts.mode = DisplayMode::OneByteDecimal,
                    b'D' => opts.mode = DisplayMode::TwoByteDecimal,
                    b'c' => opts.mode = DisplayMode::OneByteChar,
                    b'b' => opts.mode = DisplayMode::Binary,
                    b'p' => opts.mode = DisplayMode::Plain,
                    b'i' => opts.mode = DisplayMode::CInclude,
                    b'r' => opts.mode = DisplayMode::Reverse,
                    b'A' => opts.no_ascii = true,
                    b'P' => opts.no_position = true,
                    b'u' => opts.uppercase = true,
                    b'w' => opts.squeeze = true,
                    b'q' => opts.quiet = true,
                    b'W' | b'g' | b'E' | b'B' | b's' | b'n' | b'j' |
                    b'L' | b'S' | b'T' | b'm' | b'F' | b'f' => {
                        let flag = bytes[j] as char;
                        let val: String = if j + 1 < bytes.len() {
                            let v = String::from_utf8_lossy(&bytes[j+1..]).into_owned();
                            v
                        } else {
                            i += 1;
                            raw.get(i).cloned()
                                .ok_or_else(|| format!("missing argument for -{}", flag))?
                        };
                        match flag {
                            'W' => {
                                opts.width = parse_size_unsigned(&val)? as usize;
                                if opts.width == 0 { return Err("-W must be > 0".into()); }
                            }
                            'g' => {
                                opts.group = parse_size_unsigned(&val)? as usize;
                                if !matches!(opts.group, 1|2|4|8) {
                                    return Err("-g must be 1, 2, 4, or 8".into());
                                }
                            }
                            'E' => opts.endian = match val.as_str() {
                                "big"    => Endian::Big,
                                "little" => Endian::Little,
                                _ => return Err(format!("unknown endian: {}", val)),
                            },
                            'B' => opts.border = match val.as_str() {
                                "none"    => BorderStyle::None,
                                "ascii"   => BorderStyle::Ascii,
                                "unicode" => BorderStyle::Unicode,
                                _ => return Err(format!("unknown border: {}", val)),
                            },
                            's' => opts.skip   = parse_size_signed(&val)?,
                            'n' => opts.length = Some(parse_size_unsigned(&val)?),
                            'j' => opts.jump   = parse_size_signed(&val)?,
                            'L' => opts.color  = parse_color_when(&val)?,
                            'S' => opts.scheme = parse_scheme(&val)?,
                            'T' => opts.table  = parse_char_table(&val)?,
                            'm' => opts.max_lines = Some(parse_size_unsigned(&val)?),
                            'F' => {
                                if !val.contains('"') {
                                    return Err(format!("-F expects a quoted format string: {}", val));
                                }
                                opts.formats.push(val);
                            }
                            'f' => {
                                let content = std::fs::read_to_string(&val)
                                    .map_err(|e| format!("cannot read {}: {}", val, e))?;
                                for line in content.lines() {
                                    let l = line.trim();
                                    if !l.is_empty() { opts.formats.push(l.to_string()); }
                                }
                            }
                            _ => unreachable!(),
                        }
                        break;
                    }
                    _ => return Err(format!("unknown option: -{}", bytes[j] as char)),
                }
                j += 1;
            }
        } else {
            opts.files.push(arg.clone());
        }
        i += 1;
    }

    Ok(opts)
}

fn parse_color_when(s: &str) -> Result<ColorWhen, String> {
    match s {
        "auto"   => Ok(ColorWhen::Auto),
        "always" => Ok(ColorWhen::Always),
        "never"  => Ok(ColorWhen::Never),
        _ => Err(format!("unknown color mode: {}", s)),
    }
}

fn parse_scheme(s: &str) -> Result<ColorScheme, String> {
    match s {
        "default"  => Ok(ColorScheme::Default),
        "type"     => Ok(ColorScheme::Type),
        "gradient" => Ok(ColorScheme::Gradient),
        _ => Err(format!("unknown color scheme: {}", s)),
    }
}

fn parse_char_table(s: &str) -> Result<CharTable, String> {
    match s {
        "ascii"   => Ok(CharTable::Ascii),
        "default" => Ok(CharTable::Default),
        "braille" => Ok(CharTable::Braille),
        "cp437"   => Ok(CharTable::Cp437),
        "ebcdic"  => Ok(CharTable::Ebcdic),
        _ => Err(format!("unknown char table: {}", s)),
    }
}

fn parse_args() -> Result<Options, String> {
    // Prepend FASTHEX_DEFAULT_OPTS
    let mut all_args: Vec<String> = Vec::new();
    if let Ok(defaults) = env::var("FASTHEX_DEFAULT_OPTS") {
        for tok in defaults.split_ascii_whitespace() {
            all_args.push(tok.to_string());
        }
    }
    let argv: Vec<String> = env::args().skip(1).collect();
    all_args.extend(argv);

    let mut opts = parse_args_from(&all_args)?;

    if opts.mode == DisplayMode::Reverse {
        opts.reverse_jump = Some(opts.jump);
    }

    if opts.group == 1 {
        match opts.mode {
            DisplayMode::TwoByteHex | DisplayMode::TwoByteOctal |
            DisplayMode::TwoByteDecimal => opts.group = 2,
            _ => {}
        }
    }

    if opts.mode == DisplayMode::Binary { opts.width = 8; }

    if opts.mode == DisplayMode::CInclude {
        opts.include_name = opts.files.first().map(|f| {
            std::path::Path::new(f)
                .file_stem()
                .and_then(|n| n.to_str())
                .unwrap_or("data")
                .replace(|c: char| !c.is_alphanumeric(), "_")
        });
    }

    if opts.color == ColorWhen::Auto && env::var_os("NO_COLOR").is_some() {
        opts.color = ColorWhen::Never;
    }

    Ok(opts)
}

// xxd behaviour: normally 8 hex digits, grows naturally past 0xFFFFFFFF
#[inline(always)]
fn offset_len(off: u64) -> usize {
    if off <= 0xFFFF_FFFF           { 8  }
    else if off <= 0xF_FFFF_FFFF    { 9  }
    else if off <= 0xFF_FFFF_FFFF   { 10 }
    else if off <= 0xFFF_FFFF_FFFF  { 11 }
    else if off <= 0xFFFF_FFFF_FFFF { 12 }
    else if off <= 0xF_FFFF_FFFF_FFFF  { 13 }
    else if off <= 0xFF_FFFF_FFFF_FFFF { 14 }
    else if off <= 0xFFF_FFFF_FFFF_FFFF { 15 }
    else { 16 }
}

#[inline(always)]
fn write_offset(dst: &mut [u8], off: u64, dec: bool, upper: bool) -> usize {
    let hex = if upper { HEX_UPPER } else { HEX_LOWER };
    if dec {
        let s = format!("{:08}", off);
        let b = s.as_bytes();
        dst[..b.len()].copy_from_slice(b);
        b.len()
    } else {
        let len = offset_len(off);
        for k in 0..len {
            dst[len - 1 - k] = hex[((off >> (k * 4)) & 0xf) as usize];
        }
        len
    }
}

fn _char_for_byte(b: u8, table: CharTable) -> &'static str {
    match table {
        CharTable::Ascii => {
            if b >= 0x20 && b <= 0x7e { unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(&HEX_LOWER[0], 0)) } }
            else { "." }
        }
        CharTable::Default => {
            if b == 0x00 { "⋄" }
            else if b == 0x20 { " " }
            else if b >= 0x21 && b <= 0x7e { "?" } // placeholder; inline in hot path
            else { "•" }
        }
        _ => "." // placeholder; inline in hot paths
    }
}

fn braille_for_byte(b: u8) -> [u8; 3] {
    let cp = 0x2800u32 + b as u32;
    [
        0xe2,
        0xa0 | ((cp >> 6) & 0x3f) as u8,
        0x80 | (cp & 0x3f) as u8,
    ]
}

/// ANSI sequence for a byte value under the given scheme.
fn byte_ansi(b: u8, scheme: ColorScheme) -> &'static str {
    match scheme {
        ColorScheme::Default => "\x1b[32m",
        ColorScheme::Type => match b {
            0x00        => "\x1b[90m",
            0x01..=0x1f => "\x1b[33m",
            0x20        => "\x1b[36m",
            0x21..=0x7e => "\x1b[32m",
            0x7f        => "\x1b[33m",
            0x80..=0xff => "\x1b[31m",
        },
        ColorScheme::Gradient => {
            match b {
                0x00        => "\x1b[90m",
                0x01..=0x3f => "\x1b[34m",
                0x40..=0x7f => "\x1b[32m",
                0x80..=0xbf => "\x1b[33m",
                0xc0..=0xff => "\x1b[31m",
            }
        }
    }
}

const ANSI_RESET:  &str = "\x1b[0m";
const ANSI_CYAN:   &str = "\x1b[36m";
const ANSI_DIM:    &str = "\x1b[90m";

struct ZeroCopyWriter {
    pipe_r:   libc::c_int,
    pipe_w:   libc::c_int,
    stdout:   libc::c_int,
    fallback: bool,
}

impl ZeroCopyWriter {
    fn new() -> io::Result<Self> {
        #[cfg(not(target_os = "linux"))]
        {
            return Ok(Self { pipe_r: -1, pipe_w: -1, stdout: libc::STDOUT_FILENO, fallback: true });
        }
        #[cfg(target_os = "linux")]
        {
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        let ok = unsafe { libc::fstat(libc::STDOUT_FILENO, &mut st) == 0 };
        let mode = if ok { st.st_mode & libc::S_IFMT } else { 0 };
        if mode != libc::S_IFREG && mode != libc::S_IFIFO {
            return Ok(Self { pipe_r: -1, pipe_w: -1, stdout: libc::STDOUT_FILENO, fallback: true });
        }
        let mut fds = [0i32; 2];
        if unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
            return Err(io::Error::last_os_error());
        }
        unsafe { libc::fcntl(fds[1], libc::F_SETPIPE_SZ, PIPE_SIZE_HINT); }
        Ok(Self { pipe_r: fds[0], pipe_w: fds[1], stdout: libc::STDOUT_FILENO, fallback: false })
        }
    }

    fn write_chunk(&mut self, buf: &[u8]) -> io::Result<()> {
        if self.fallback { return self.write_fallback(buf); }
        #[cfg(target_os = "linux")]
        { return unsafe { self.write_zero_copy(buf) }; }
        #[cfg(not(target_os = "linux"))]
        { return self.write_fallback(buf); }
    }

    #[cfg(target_os = "linux")]
    unsafe fn write_zero_copy(&mut self, buf: &[u8]) -> io::Result<()> { unsafe {
        let mut pos = 0usize;
        while pos < buf.len() {
            let iov = libc::iovec {
                iov_base: buf.as_ptr().add(pos) as *mut libc::c_void,
                iov_len:  buf.len() - pos,
            };
            let n = libc::vmsplice(self.pipe_w, &iov, 1, libc::SPLICE_F_GIFT);
            if n < 0 {
                let e = io::Error::last_os_error();
                if e.raw_os_error() == Some(libc::EINVAL) || e.raw_os_error() == Some(libc::ENOSYS) {
                    return self.fallback_write(&buf[pos..]);
                }
                return Err(e);
            }
            if n == 0 {
                return Err(io::Error::new(io::ErrorKind::WriteZero, "vmsplice returned 0"));
            }
            let n = n as usize;
            pos += n;
            if let Err(e) = self.splice_out(n) {
                if e.raw_os_error() == Some(libc::EINVAL) {
                    return self.fallback_write(&buf[pos..]);
                }
                return Err(e);
            }
        }
        Ok(())
    }}

    #[cfg(target_os = "linux")]
    unsafe fn splice_out(&mut self, len: usize) -> io::Result<()> { unsafe {
        let mut remain = len;
        while remain > 0 {
            let n = libc::splice(self.pipe_r, std::ptr::null_mut(),
                                 self.stdout, std::ptr::null_mut(),
                                 remain, libc::SPLICE_F_MOVE);
            if n < 0 { return Err(io::Error::last_os_error()); }
            if n == 0 { return Err(io::Error::new(io::ErrorKind::BrokenPipe, "splice ended early")); }
            remain -= n as usize;
        }
        Ok(())
    }}

    fn fallback_write(&mut self, remainder: &[u8]) -> io::Result<()> {
        self.fallback = true;
        let fl = unsafe { libc::fcntl(self.pipe_r, libc::F_GETFL) };
        if fl >= 0 {
            unsafe { libc::fcntl(self.pipe_r, libc::F_SETFL, fl | libc::O_NONBLOCK); }
        }
        let mut tmp = vec![0u8; 65536];
        loop {
            let n = unsafe { libc::read(self.pipe_r, tmp.as_mut_ptr() as *mut libc::c_void, tmp.len()) };
            if n > 0 {
                io::stdout().lock().write_all(&tmp[..n as usize])?;
            } else if n < 0 {
                let e = io::Error::last_os_error();
                if e.raw_os_error() == Some(libc::EINTR) { continue; }
                break;
            } else {
                break;
            }
        }
        self.write_fallback(remainder)
    }

    fn write_fallback(&self, buf: &[u8]) -> io::Result<()> {
        io::stdout().lock().write_all(buf)
    }
}

impl Drop for ZeroCopyWriter {
    fn drop(&mut self) {
        if self.pipe_r >= 0 { unsafe { libc::close(self.pipe_r); } }
        if self.pipe_w >= 0 { unsafe { libc::close(self.pipe_w); } }
    }
}

struct HexWin {
    off:   usize,
    idx:   [i8; 16],
    sp:    [u8; 16],
    hi:    bool,
    block: usize,
}

struct ConstBlock {
    off: usize,
    len: u8,
    val: u64,
}

#[derive(Clone, Copy)]
struct CanonKernel {
    idx_a:    [[i8; 16]; 3],
    idx_b:    [[i8; 16]; 3],
    sp:       [[u8; 16]; 3],
    w3_idx:   [i8; 16],
    w3_sp:    [u8; 16],
    tu_idx:   [i8; 16],
    tu_sp:    [u8; 16],
    nl:       u8,
    has_ascii: bool,
    emitted:  usize,
}

struct ColoredWin {
    idx_c:  [[i8; 16]; 5],
    idx_ch: [i8; 16],
    sp:     [u8; 16],
}

struct ColoredPanel {
    panel_off: usize,
    wins:      [ColoredWin; 10],
}

struct RowLayout {
    emitted:       usize,
    windows:       Vec<HexWin>,
    ascii_windows: Vec<HexWin>,
    ascii_off:     usize,
    ascii_run:     usize,
    esc_mask:      Option<[u8; 16]>,
    consts:        Vec<ConstBlock>,
    all_fit:       bool,
    fast:          Option<CanonKernel>,
    colored:       Option<ColoredPanel>,
}

#[inline(always)]
unsafe fn store_block(dst: *mut u8, b: &ConstBlock) { unsafe {
    match b.len {
        1 => *dst.add(b.off) = b.val as u8,
        2 => *(dst.add(b.off) as *mut u16) = b.val as u16,
        4 => *(dst.add(b.off) as *mut u32) = b.val as u32,
        _ => *(dst.add(b.off) as *mut u64) = b.val,
    }
}}

struct RowCore {
    opts:                 Options,
    width:                usize,
    blocks:               usize,
    fixed_prefix:         bool,
    prefix_fixed:         usize,
    pos_w:                usize,
    bar:                  [u8; 3],
    bar_len:              usize,
    full:                 RowLayout,
    rev:                  Option<[i8; 16]>,
    tail_kind:            LayoutKind,
    do_color:             bool,
}

struct RowCfg {
    core:  RowCore,
    tails: RefCell<HashMap<usize, RowLayout>>,
}

fn field_len_for(opts: &Options, off: u64) -> usize {
    if opts.offset_dec {
        let mut n = 1u32;
        let mut x = off;
        while x >= 10 { x /= 10; n += 1; }
        (n as usize).max(8)
    } else {
        offset_len(off)
    }
}

#[derive(Clone, Copy, PartialEq)]
enum LayoutKind { Generic, OutputLine, Scalar }

fn format_scalar_row(dst: &mut Vec<u8>, src: &[u8], off: u64, no_ascii: bool) {
    let n     = src.len();
    let width = 16;
    let half  = width / 2;
    let mut tmp = [0u8; 20];
    let olen = write_offset(&mut tmp, off, false, false);
    dst.extend_from_slice(&tmp[..olen]);
    dst.push(b':');
    dst.push(b' ');
    for i in 0..width {
        if i > 0 && i % half == 0 { dst.push(b' '); }
        if i < n {
            let b = src[i];
            dst.push(HEX_LOWER[(b >> 4) as usize]);
            dst.push(HEX_LOWER[(b & 0xf) as usize]);
            if i < width - 1 || !no_ascii { dst.push(b' '); }
        } else {
            dst.push(b' ');
            dst.push(b' ');
            if i < width - 1 || !no_ascii { dst.push(b' '); }
        }
    }
    if no_ascii {
        dst.push(b'\n');
        return;
    }
    dst.push(b'|');
    for i in 0..n {
        let b = src[i];
        dst.push(if b >= 0x20 && b <= 0x7e { b } else { b'.' });
    }
    for _ in n..width {
        dst.push(b' ');
    }
    dst.push(b'|');
    dst.push(b'\n');
}

fn build_layout(opts: &Options, n: usize, kind: LayoutKind) -> RowLayout {
    let mut derive_opts = opts.clone();
    derive_opts.endian = Endian::Big;
    let mut r1 = Vec::new();
    let mut r2 = Vec::new();
    match kind {
        LayoutKind::OutputLine if !output_line_diverts(opts) => {
            let dc = use_color(opts);
            output_line(&mut r1, &vec![0x55; n], 0, &derive_opts, dc, 0, 0).unwrap();
            output_line(&mut r2, &vec![0xAA; n], 1, &derive_opts, dc, 0, 0).unwrap();
        }
        LayoutKind::Scalar => {
            format_scalar_row(&mut r1, &vec![0x55; n], 0, opts.no_ascii);
            format_scalar_row(&mut r2, &vec![0xAA; n], 1, opts.no_ascii);
        }
        _ => {
            format_row_generic(&mut r1, &vec![0x55; n], 0, &derive_opts);
            format_row_generic(&mut r2, &vec![0xAA; n], 1, &derive_opts);
        }
    }
    let bar = match opts.border {
        BorderStyle::None    => "",
        BorderStyle::Ascii   => "|",
        BorderStyle::Unicode => "│",
    };
    let pos_w = (if opts.offset_dec { 20 } else { offset_len(u64::MAX) }) + 1;
    let prefix_len = if opts.border != BorderStyle::None {
        bar.len() + if opts.no_position { 0 } else { pos_w }
    } else if opts.no_position {
        0
    } else {
        field_len_for(opts, 0) + if opts.minimal { 1 } else if use_color(opts) { 11 } else { 2 }
    };
    let mut suffix = r1[prefix_len..].to_vec();
    for i in 0..suffix.len() {
        if r1[prefix_len + i] != r2[prefix_len + i] {
            suffix[i] = 0xFF;
        }
    }
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < suffix.len() {
        if suffix[i] == 0xFF {
            let s = i;
            while i < suffix.len() && suffix[i] == 0xFF { i += 1; }
            runs.push((s, i - s));
        } else {
            i += 1;
        }
    }
    let mut hex_pairs: Vec<(usize, usize)> = Vec::new();
    let mut tail_runs: Vec<(usize, usize)> = Vec::new();
    for &(s, l) in &runs {
        let mut consumed = 0usize;
        while consumed + 1 < l && hex_pairs.len() < n {
            hex_pairs.push((s + consumed, s + consumed + 1));
            consumed += 2;
        }
        if consumed < l {
            tail_runs.push((s + consumed, l - consumed));
        }
    }
    let mut ascii_off = 0usize;
    let mut ascii_run = 0usize;
    let mut ascii_pairs: Vec<(usize, usize)> = Vec::new();
    match opts.table {
        CharTable::Braille => {
            for &(s, l) in &tail_runs {
                if l == 2 {
                    if ascii_pairs.is_empty() { ascii_off = s; }
                    ascii_pairs.push((s, s + 1));
                }
            }
        }
        _ => {
            for &(s, l) in &tail_runs {
                if ascii_run == 0 { ascii_off = s; }
                ascii_run = s + l - ascii_off;
            }
        }
    }
    let colored_ok = use_color(opts)
        && opts.scheme == ColorScheme::Default
        && opts.table == CharTable::Ascii
        && !opts.no_ascii
        && opts.border == BorderStyle::None
        && opts.mode == DisplayMode::Canonical;
    let mut colored = None;
    if colored_ok && hex_pairs.len() == n && !hex_pairs.is_empty() && !tail_runs.is_empty()
        && tail_runs[0].0 >= 2 {
        let panel_off = tail_runs[0].0 - 2;
        let mut wins: [ColoredWin; 10] = std::array::from_fn(|_| ColoredWin {
            idx_c: [[-1i8; 16]; 5],
            idx_ch: [-1i8; 16],
            sp: [0u8; 16],
        });
        const RESET: [u8; 4] = [0x1b, 0x5b, 0x30, 0x6d];
        for w in 0..10 {
            for k in 0..16 {
                let ppos = w * 16 + k;
                let j = ppos / 10;
                let c = ppos % 10;
                if j < n {
                    if c < 5 {
                        wins[w].idx_c[c][k] = j as i8;
                    } else if c == 5 {
                        wins[w].idx_ch[k] = j as i8;
                    } else {
                        wins[w].sp[k] = RESET[c - 6];
                    }
                } else {
                    wins[w].sp[k] = b' ';
                }
            }
        }
        let end = panel_off + n * 10;
        if suffix.len() < end {
            suffix.resize(end, b' ');
        }
        colored = Some(ColoredPanel { panel_off, wins });
    }
    let build_windows = |pairs: &[(usize, usize)], suffix: &[u8]| -> Vec<HexWin> {
        let mut out = Vec::new();
        let npairs = pairs.len();
        let mut q0 = 0usize;
        while q0 < npairs {
            let start = pairs[q0].0;
            let half = (q0 % 16) / 8;
            let mut q1 = q0;
            while q1 + 1 < npairs && (q1 + 1) / 16 == q0 / 16
                && ((q1 + 1) % 16) / 8 == half
                && pairs[q1 + 1].1 + 1 - start <= 16 {
                q1 += 1;
            }
            let mut idx = [-1i8; 16];
            let mut sp = [0u8; 16];
            for k in 0..16 {
                let pos = start + k;
                let mut hit = None;
                for j in q0..=q1 {
                    for c in 0..2 {
                        let slot = if c == 0 { pairs[j].0 } else { pairs[j].1 };
                        if slot == pos {
                            hit = Some((2 * (j % 8) + c) as i8);
                        }
                    }
                }
                match hit {
                    Some(v) => idx[k] = v,
                    None => sp[k] = if pos < suffix.len() { suffix[pos] } else { b' ' },
                }
            }
            out.push(HexWin { off: start, idx, sp, hi: half == 1, block: q0 / 16 });
            q0 = q1 + 1;
        }
        out
    };
    let windows = build_windows(&hex_pairs, &suffix);
    let ascii_windows = if opts.table == CharTable::Braille {
        build_windows(&ascii_pairs, &suffix)
    } else { Vec::new() };
    let emitted = suffix.len();
    let mut covered = vec![false; emitted];
    for w in &windows {
        for k in w.off..(w.off + 16).min(emitted) { covered[k] = true; }
    }
    for w in &ascii_windows {
        for k in w.off..(w.off + 16).min(emitted) { covered[k] = true; }
    }
    if opts.table == CharTable::Ascii {
        for k in ascii_off..(ascii_off + ascii_run).min(emitted) { covered[k] = true; }
    }
    if let Some(cp) = &colored {
        for k in cp.panel_off..(cp.panel_off + n * 10).min(emitted) { covered[k] = true; }
    }
    let mut consts: Vec<ConstBlock> = Vec::new();
    let mut i = 0;
    while i < emitted {
        if !covered[i] {
            let s = i;
            while i < emitted && !covered[i] { i += 1; }
            let mut p = s;
            while p < i {
                let rem = i - p;
                let l = if rem >= 8 { 8 } else if rem >= 4 { 4 } else if rem >= 2 { 2 } else { 1 };
                let mut val = 0u64;
                for k in 0..l {
                    val |= (suffix[p + k] as u64) << (k * 8);
                }
                consts.push(ConstBlock { off: p, len: l as u8, val });
                p += l;
            }
        } else {
            i += 1;
        }
    }
    let all_fit = windows.iter().all(|w| w.off + 16 <= emitted)
        && ascii_windows.iter().all(|w| w.off + 16 <= emitted);
    let canonical_shape = opts.mode == DisplayMode::Canonical
        && opts.width == 16
        && (ascii_run >= 16 || (opts.no_ascii && ascii_run == 0))
        && hex_pairs.len() == 16
        && (0..16).all(|j| {
            let o = if j >= 8 { 1 } else { 0 };
            hex_pairs[j] == (3 * j + o, 3 * j + 1 + o)
        });
    let fast = if canonical_shape && opts.table == CharTable::Ascii && opts.endian == Endian::Big
        && opts.border == BorderStyle::None && colored.is_none() {
        let mut k = CanonKernel {
            idx_a:    [[-1i8; 16]; 3],
            idx_b:    [[-1i8; 16]; 3],
            sp:       [[0u8; 16]; 3],
            w3_idx:   [-1i8; 16],
            w3_sp:    [0u8; 16],
            tu_idx:   [-1i8; 16],
            tu_sp:    [0u8; 16],
            nl:       0,
            has_ascii: false,
            emitted:   0,
        };
        let mut ok = true;
        for (bi, base) in [0usize, 16, 32].iter().enumerate() {
            for kk in 0..16 {
                let pos = base + kk;
                if let Some(ci) = hex_pairs.iter().position(|&(a, b)| a == pos || b == pos) {
                    let c = if hex_pairs[ci].0 == pos { 2 * ci } else { 2 * ci + 1 };
                    if c < 16 { k.idx_a[bi][kk] = c as i8; } else { k.idx_b[bi][kk] = (c - 16) as i8; }
                } else if pos < suffix.len() {
                    if suffix[pos] != 0xFF { k.sp[bi][kk] = suffix[pos]; } else { ok = false; }
                }
            }
        }
        k.emitted = emitted;
        k.has_ascii = ascii_run >= 16;
        if k.has_ascii {
            if emitted >= 69 {
                for kk in 0..16 {
                    let pos = 48 + kk;
                    if pos >= ascii_off && pos < ascii_off + ascii_run && pos - ascii_off < 16 {
                        k.w3_idx[kk] = (pos - ascii_off) as i8;
                    } else if pos < suffix.len() && suffix[pos] != 0xFF {
                        k.w3_sp[kk] = suffix[pos];
                    } else {
                        ok = false;
                    }
                }
                for kk in 0..4 {
                    let pos = 64 + kk;
                    if pos >= ascii_off && pos < ascii_off + ascii_run && pos - ascii_off < 16 {
                        k.tu_idx[kk] = (pos - ascii_off) as i8;
                    } else if pos < suffix.len() && suffix[pos] != 0xFF {
                        k.tu_sp[kk] = suffix[pos];
                    } else {
                        ok = false;
                    }
                }
                k.nl = suffix[68];
            } else {
                ok = false;
            }
        } else if emitted >= 48 {
            for kk in 0..16 {
                let pos = 48 + kk;
                if pos < suffix.len() {
                    if suffix[pos] != 0xFF { k.w3_sp[kk] = suffix[pos]; } else { ok = false; }
                }
            }
        } else {
            ok = false;
        }
        if ok { Some(k) } else { None }
    } else { None };
    let esc_mask = if opts.endian == Endian::Little && opts.group > 1
        && opts.mode == DisplayMode::Canonical {
        let g = opts.group;
        let rem = n % g;
        if rem != 0 {
            let grp_start = n - rem;
            let last_start = ((n + 15) / 16 - 1) * 16;
            let mut m = [0xFFu8; 16];
            for j in 0..16 {
                let i = last_start + j;
                if i >= grp_start && i < n && i % g <= g - 1 - rem {
                    m[j] = 0x00;
                }
            }
            Some(m)
        } else { None }
    } else { None };
    RowLayout { emitted, windows, ascii_windows, ascii_off, ascii_run, esc_mask, consts, all_fit, fast, colored }
}

impl RowCfg {
    fn new(opts: &Options, full_kind: LayoutKind, tail_kind: LayoutKind) -> RowCfg {
        let width  = opts.width;
        let blocks = (width + 15) / 16;
        let full   = build_layout(opts, width, full_kind);
        let pos_w  = (if opts.offset_dec { 20 } else { offset_len(u64::MAX) }) + 1;
        let bar = match opts.border {
            BorderStyle::None    => [0u8; 3],
            BorderStyle::Ascii   => [b'|', 0, 0],
            BorderStyle::Unicode => [0xe2, 0x94, 0x82],
        };
        let bar_len = match opts.border {
            BorderStyle::None => 0, BorderStyle::Ascii => 1, BorderStyle::Unicode => 3,
        };
        let fixed_prefix = opts.border != BorderStyle::None || opts.no_position;
        let prefix_fixed = if fixed_prefix {
            bar_len + if opts.no_position { 0 } else { pos_w }
        } else { 0 };
        let rev = if opts.endian == Endian::Little {
            let g = match opts.mode {
                DisplayMode::TwoByteHex => opts.group.max(2),
                _ => opts.group,
            };
            if opts.mode == DisplayMode::Canonical && g <= 1 {
                None
            } else if opts.mode != DisplayMode::Canonical && opts.mode != DisplayMode::TwoByteHex {
                None
            } else {
                let mut m = [0i8; 16];
                for j in 0..16 {
                    m[j] = ((j / g) * g + (g - 1 - j % g)) as i8;
                }
                Some(m)
            }
        } else { None };
        RowCfg {
            core: RowCore {
                opts: opts.clone(), width, blocks,
                fixed_prefix, prefix_fixed, pos_w, bar, bar_len, full, rev,
                tail_kind, do_color: use_color(opts),
            },
            tails: RefCell::new(HashMap::new()),
        }
    }

    fn layout(&self, n: usize) -> &RowLayout {
        if n == self.core.width {
            &self.core.full
        } else {
            let mut t = self.tails.borrow_mut();
            t.entry(n).or_insert_with(|| build_layout(&self.core.opts, n, self.core.tail_kind));
            unsafe { &*(t.get(&n).unwrap() as *const RowLayout) }
        }
    }
}

impl RowCore {
    fn prefix_len(&self, field_len: usize) -> usize {
        if self.fixed_prefix { self.prefix_fixed } else {
            field_len + if self.opts.minimal { 1 } else if self.do_color { 11 } else { 2 }
        }
    }
}

fn can_pair(core: &RowCore) -> bool {
    core.width == 16
        && core.rev.is_none()
        && core.full.windows.len() == 4
        && (core.full.ascii_run >= 16 || core.full.fast.is_some())
        && core.full.ascii_windows.is_empty()
        && core.opts.table == CharTable::Ascii
        && core.full.colored.is_none()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn hex_offsets_4(off: u64, lut: &[u8; 16]) -> [u64; 4] { unsafe {
    let base  = _mm_set1_epi32(off as i32);
    let delta = _mm_setr_epi32(0, 16, 32, 48);
    let x = _mm_add_epi32(base, delta);
    let x = _mm_shuffle_epi8(x, _mm_setr_epi8(3,2,1,0, 7,6,5,4, 11,10,9,8, 15,14,13,12));
    let m = _mm_set1_epi8(0x0f);
    let lo = _mm_and_si128(x, m);
    let hi = _mm_and_si128(_mm_srli_epi16(x, 4), m);
    let lutv = _mm_loadu_si128(lut.as_ptr() as *const __m128i);
    let hlo = _mm_shuffle_epi8(lutv, lo);
    let hhi = _mm_shuffle_epi8(lutv, hi);
    let plo = _mm_unpacklo_epi8(hhi, hlo);
    let phi = _mm_unpackhi_epi8(hhi, hlo);
    [
        _mm_cvtsi128_si64(plo) as u64,
        _mm_cvtsi128_si64(_mm_srli_si128(plo, 8)) as u64,
        _mm_cvtsi128_si64(phi) as u64,
        _mm_cvtsi128_si64(_mm_srli_si128(phi, 8)) as u64,
    ]
}}

#[inline(always)]
unsafe fn write_hex_offset(dst: *mut u8, off: u64, lut: &[u8; 16]) -> usize { unsafe {
    if off <= 0xFFFF_FFFF {
        for k in 0..8 {
            *dst.add(7 - k) = lut[((off >> (k * 4)) & 0xf) as usize];
        }
        8
    } else {
        let len = offset_len(off);
        for k in 0..len {
            *dst.add(len - 1 - k) = lut[((off >> (k * 4)) & 0xf) as usize];
        }
        len
    }
}}

#[inline(always)]
unsafe fn write_prefix(dst: *mut u8, off: u64, core: &RowCore) -> usize { unsafe {
    let mut p = dst;
    let lut = if core.opts.uppercase { HEX_UPPER } else { HEX_LOWER };
    if core.opts.border != BorderStyle::None {
        if !core.opts.no_position {
            std::ptr::copy_nonoverlapping(core.bar.as_ptr(), p, core.bar_len);
            p = p.add(core.bar_len);
            let olen = if core.opts.offset_dec {
                let mut tmp = [0u8; 20];
                let olen = write_offset(&mut tmp, off, true, false);
                std::ptr::copy_nonoverlapping(tmp.as_ptr(), p, olen);
                olen
            } else {
                write_hex_offset(p, off, lut)
            };
            p = p.add(olen);
            *p = b':';
            p = p.add(1);
            let pad = core.pos_w.saturating_sub(olen + 1);
            for _ in 0..pad {
                *p = b' ';
                p = p.add(1);
            }
        } else {
            std::ptr::copy_nonoverlapping(core.bar.as_ptr(), p, core.bar_len);
            p = p.add(core.bar_len);
        }
    } else if !core.opts.no_position {
        if core.opts.minimal {
            if core.opts.offset_dec {
                let mut tmp = [0u8; 20];
                let olen = write_offset(&mut tmp, off, true, false);
                std::ptr::copy_nonoverlapping(tmp.as_ptr(), p, olen);
                p = p.add(olen);
            } else {
                p = p.add(write_hex_offset(p, off, lut));
            }
            *p = b' ';
            p = p.add(1);
        } else if core.do_color {
            std::ptr::copy_nonoverlapping(b"\x1b[36m".as_ptr(), p, 5);
            p = p.add(5);
            if core.opts.offset_dec {
                let mut tmp = [0u8; 20];
                let olen = write_offset(&mut tmp, off, true, false);
                std::ptr::copy_nonoverlapping(tmp.as_ptr(), p, olen);
                p = p.add(olen);
            } else {
                p = p.add(write_hex_offset(p, off, lut));
            }
            *p = b':';
            std::ptr::copy_nonoverlapping(b"\x1b[0m".as_ptr(), p.add(1), 4);
            *p.add(5) = b' ';
            p = p.add(6);
        } else {
            if core.opts.offset_dec {
                let mut tmp = [0u8; 20];
                let olen = write_offset(&mut tmp, off, true, false);
                std::ptr::copy_nonoverlapping(tmp.as_ptr(), p, olen);
                p = p.add(olen);
            } else {
                p = p.add(write_hex_offset(p, off, lut));
            }
            *p = b':';
            *p.add(1) = b' ';
            p = p.add(2);
        }
    }
    p as usize - dst as usize
}}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3,sse4.1")]
unsafe fn format_row(dst: *mut u8, src: *const u8, off: u64, n: usize,
                     core: &RowCore, layout: &RowLayout) -> usize { unsafe {
    let blocks = (n + 15) / 16;
    let p = dst.add(write_prefix(dst, off, core));
    for cb in &layout.consts {
        store_block(p, cb);
    }
    for b in 0..blocks {
        let raw = _mm_loadu_si128(src.add(b * 16) as *const __m128i);
        let input = match &core.rev {
            Some(m) => {
                let rev = _mm_shuffle_epi8(raw, _mm_loadu_si128(m.as_ptr() as *const __m128i));
                if b == blocks - 1 {
                    if let Some(em) = &layout.esc_mask {
                        _mm_and_si128(rev, _mm_loadu_si128(em.as_ptr() as *const __m128i))
                    } else { rev }
                } else { rev }
            }
            None => raw,
        };
        let m0f = _mm_set1_epi8(0x0f);
        let lo  = _mm_and_si128(input, m0f);
        let hi  = _mm_and_si128(_mm_srli_epi16(input, 4), m0f);
        let lut = if core.opts.uppercase {
            _mm_loadu_si128(HEX_UPPER.as_ptr() as *const __m128i)
        } else {
            _mm_loadu_si128(HEX_LOWER.as_ptr() as *const __m128i)
        };
        let hlo = _mm_shuffle_epi8(lut, lo);
        let hhi = _mm_shuffle_epi8(lut, hi);
        let plo = _mm_unpacklo_epi8(hhi, hlo);
        let phi = _mm_unpackhi_epi8(hhi, hlo);
        if layout.all_fit {
            for wi in 0..layout.windows.len() {
                if layout.windows[wi].block != b { continue; }
                let w = &layout.windows[wi];
                let half = if w.hi { phi } else { plo };
                let v = _mm_or_si128(
                    _mm_shuffle_epi8(half, _mm_loadu_si128(w.idx.as_ptr() as *const __m128i)),
                    _mm_loadu_si128(w.sp.as_ptr() as *const __m128i));
                _mm_storeu_si128(p.add(w.off) as *mut __m128i, v);
            }
        } else {
            for wi in 0..layout.windows.len() {
                if layout.windows[wi].block != b { continue; }
                let w = &layout.windows[wi];
                let half = if w.hi { phi } else { plo };
                let v = _mm_or_si128(
                    _mm_shuffle_epi8(half, _mm_loadu_si128(w.idx.as_ptr() as *const __m128i)),
                    _mm_loadu_si128(w.sp.as_ptr() as *const __m128i));
                if w.off + 16 <= layout.emitted {
                    _mm_storeu_si128(p.add(w.off) as *mut __m128i, v);
                } else {
                    let mut tmp = [0u8; 16];
                    _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, v);
                    std::ptr::copy_nonoverlapping(tmp.as_ptr(), p.add(w.off), layout.emitted - w.off);
                }
            }
        }
    }
    if let Some(cp) = &layout.colored {
        const GREEN: [u8; 5] = [0x1b, 0x5b, 0x33, 0x32, 0x6d];
        const DIM: [u8; 5] = [0x1b, 0x5b, 0x39, 0x30, 0x6d];
        for b in 0..blocks {
            let raw = _mm_loadu_si128(src.add(b * 16) as *const __m128i);
            let pr = _mm_and_si128(
                _mm_cmpgt_epi8(raw, _mm_set1_epi8(0x1f)),
                _mm_cmpgt_epi8(_mm_set1_epi8(0x7f), raw));
            let chars = _mm_blendv_epi8(_mm_set1_epi8(b'.' as i8), raw, pr);
            let mut planes = [_mm_setzero_si128(); 5];
            for c in 0..5 {
                planes[c] = _mm_blendv_epi8(
                    _mm_set1_epi8(DIM[c] as i8), _mm_set1_epi8(GREEN[c] as i8), pr);
            }
            let panel_end = cp.panel_off + n * 10;
            let chars_here = (n - b * 16).min(16);
            let nwin = (chars_here * 10 + 15) / 16;
            for w in 0..nwin {
                let win = &cp.wins[w];
                let mut v = _mm_loadu_si128(win.sp.as_ptr() as *const __m128i);
                for c in 0..5 {
                    v = _mm_or_si128(v, _mm_shuffle_epi8(
                        planes[c], _mm_loadu_si128(win.idx_c[c].as_ptr() as *const __m128i)));
                }
                v = _mm_or_si128(v, _mm_shuffle_epi8(
                    chars, _mm_loadu_si128(win.idx_ch.as_ptr() as *const __m128i)));
                let off = cp.panel_off + b * 160 + w * 16;
                if off + 16 <= panel_end {
                    _mm_storeu_si128(p.add(off) as *mut __m128i, v);
                } else {
                    let mut tmp = [0u8; 16];
                    _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, v);
                    std::ptr::copy_nonoverlapping(tmp.as_ptr(), p.add(off), panel_end - off);
                }
            }
        }
    } else {
    match core.opts.table {
        CharTable::Ascii => {
            let mut done = 0usize;
            for b in 0..blocks {
                let avail = layout.ascii_run.saturating_sub(done);
                if avail == 0 { break; }
                let raw = _mm_loadu_si128(src.add(b * 16) as *const __m128i);
                if avail >= 16 {
                    let pr = _mm_and_si128(
                        _mm_cmpgt_epi8(raw, _mm_set1_epi8(0x1f)),
                        _mm_cmpgt_epi8(_mm_set1_epi8(0x7f), raw));
                    let asc = _mm_blendv_epi8(_mm_set1_epi8(b'.' as i8), raw, pr);
                    _mm_storeu_si128(p.add(layout.ascii_off + done) as *mut __m128i, asc);
                    done += 16;
                } else {
                    let d = p.add(layout.ascii_off + done);
                    for j in 0..avail {
                        let c = *src.add(b * 16 + j);
                        *d.add(j) = if (0x20..=0x7e).contains(&c) { c } else { b'.' };
                    }
                    done += avail;
                }
            }
        }
        CharTable::Braille => {
            for b in 0..blocks {
                if b * 4 >= layout.ascii_windows.len() { break; }
                let raw = _mm_loadu_si128(src.add(b * 16) as *const __m128i);
                let v1 = _mm_add_epi8(
                    _mm_and_si128(_mm_srli_epi16(raw, 6), _mm_set1_epi8(0x03)),
                    _mm_set1_epi8(0xa0u8 as i8));
                let v2 = _mm_or_si128(_mm_and_si128(raw, _mm_set1_epi8(0x3f)),
                                      _mm_set1_epi8(0x80u8 as i8));
                let plo = _mm_unpacklo_epi8(v1, v2);
                let phi = _mm_unpackhi_epi8(v1, v2);
                for q in 0..4 {
                    let wi = b * 4 + q;
                    if wi >= layout.ascii_windows.len() { break; }
                    let w = &layout.ascii_windows[wi];
                    let half = if q < 2 { plo } else { phi };
                    let v = _mm_or_si128(
                        _mm_shuffle_epi8(half, _mm_loadu_si128(w.idx.as_ptr() as *const __m128i)),
                        _mm_loadu_si128(w.sp.as_ptr() as *const __m128i));
                    if w.off + 16 <= layout.emitted {
                        _mm_storeu_si128(p.add(w.off) as *mut __m128i, v);
                    } else {
                        let mut tmp = [0u8; 16];
                        _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, v);
                        std::ptr::copy_nonoverlapping(tmp.as_ptr(), p.add(w.off), layout.emitted - w.off);
                    }
                }
            }
        }
        _ => {}
    }
    }
    (p as usize - dst as usize) + layout.emitted
}}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,ssse3,sse4.1")]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,ssse3,sse4.1")]
unsafe fn format_pair(dst: *mut u8, src: *const u8, off: u64, o8: [u64; 2], fast_off: bool,
                      idx: &[[i8; 16]; 4], sp: &[[u8; 16]; 4],
                      core: &RowCore, layout: &RowLayout, row_len: usize) -> usize { unsafe {
    let raw = _mm256_loadu_si256(src as *const __m256i);
    let input = match &core.rev {
        Some(m) => _mm256_shuffle_epi8(
            raw, _mm256_broadcastsi128_si256(_mm_loadu_si128(m.as_ptr() as *const __m128i))),
        None => raw,
    };
    let m0f = _mm256_set1_epi8(0x0f);
    let lo  = _mm256_and_si256(input, m0f);
    let hi  = _mm256_and_si256(_mm256_srli_epi16(input, 4), m0f);
    let lut = if core.opts.uppercase {
        _mm256_broadcastsi128_si256(_mm_loadu_si128(HEX_UPPER.as_ptr() as *const __m128i))
    } else {
        _mm256_broadcastsi128_si256(_mm_loadu_si128(HEX_LOWER.as_ptr() as *const __m128i))
    };
    let hlo = _mm256_shuffle_epi8(lut, lo);
    let hhi = _mm256_shuffle_epi8(lut, hi);
    let plo = _mm256_unpacklo_epi8(hhi, hlo);
    let phi = _mm256_unpackhi_epi8(hhi, hlo);
    let pr = _mm256_and_si256(
        _mm256_cmpgt_epi8(raw, _mm256_set1_epi8(0x1f)),
        _mm256_cmpgt_epi8(_mm256_set1_epi8(0x7f), raw));
    let asc = _mm256_blendv_epi8(_mm256_set1_epi8(b'.' as i8), raw, pr);
    for r in 0..2 {
        let d = dst.add(r * row_len);
        let p = if fast_off {
            std::ptr::write_unaligned(d as *mut u64, o8[r]);
            *d.add(8) = b':';
            *d.add(9) = b' ';
            d.add(10)
        } else {
            d.add(write_prefix(d, off.wrapping_add(r as u64 * 16), core))
        };
        for cb in &layout.consts {
            store_block(p, cb);
        }
        let plo_r = if r == 0 { _mm256_castsi256_si128(plo) } else { _mm256_extracti128_si256(plo, 1) };
        let phi_r = if r == 0 { _mm256_castsi256_si128(phi) } else { _mm256_extracti128_si256(phi, 1) };
        if layout.all_fit && layout.windows.len() == 4 {
            for q in 0..4 {
                let off_q = layout.windows[q].off;
                let half = if q < 2 { plo_r } else { phi_r };
                let v = _mm_or_si128(
                    _mm_shuffle_epi8(half, _mm_loadu_si128(idx[q].as_ptr() as *const __m128i)),
                    _mm_loadu_si128(sp[q].as_ptr() as *const __m128i));
                _mm_storeu_si128(p.add(off_q) as *mut __m128i, v);
            }
        } else {
            for q in 0..4 {
                let off_q = layout.windows[q].off;
                let half = if q < 2 { plo_r } else { phi_r };
                let v = _mm_or_si128(
                    _mm_shuffle_epi8(half, _mm_loadu_si128(idx[q].as_ptr() as *const __m128i)),
                    _mm_loadu_si128(sp[q].as_ptr() as *const __m128i));
                if off_q + 16 <= layout.emitted {
                    _mm_storeu_si128(p.add(off_q) as *mut __m128i, v);
                } else {
                    let mut tmp = [0u8; 16];
                    _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, v);
                    std::ptr::copy_nonoverlapping(tmp.as_ptr(), p.add(off_q), layout.emitted - off_q);
                }
            }
        }
        let asc_r = if r == 0 { _mm256_castsi256_si128(asc) } else { _mm256_extracti128_si256(asc, 1) };
        if layout.ascii_run >= 16 {
            _mm_storeu_si128(p.add(layout.ascii_off) as *mut __m128i, asc_r);
        }
    }
    row_len * 2
}}

fn pair_masks(layout: &RowLayout) -> ([[i8; 16]; 4], [[u8; 16]; 4]) {
    let mut idx = [[0i8; 16]; 4];
    let mut sp  = [[0u8; 16]; 4];
    for q in 0..4 {
        idx[q] = layout.windows[q].idx;
        sp[q]  = layout.windows[q].sp;
    }
    (idx, sp)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn format_four_rows(dst: *mut u8, src: *const u8, off: u64, o8: [u64; 4], fast_off: bool,
                           idx: &[[i8; 16]; 4], sp: &[[u8; 16]; 4],
                           core: &RowCore, layout: &RowLayout, row_len: usize) -> usize { unsafe {
    let raw = _mm512_loadu_si512(src as *const _);
    let input = match &core.rev {
        Some(m) => _mm512_shuffle_epi8(
            raw, _mm512_broadcast_i32x4(_mm_loadu_si128(m.as_ptr() as *const __m128i))),
        None => raw,
    };
    let m0f = _mm512_set1_epi8(0x0f);
    let lo  = _mm512_and_si512(input, m0f);
    let hi  = _mm512_and_si512(_mm512_srli_epi16(input, 4), m0f);
    let lut = if core.opts.uppercase {
        _mm512_broadcast_i32x4(_mm_loadu_si128(HEX_UPPER.as_ptr() as *const __m128i))
    } else {
        _mm512_broadcast_i32x4(_mm_loadu_si128(HEX_LOWER.as_ptr() as *const __m128i))
    };
    let hlo = _mm512_shuffle_epi8(lut, lo);
    let hhi = _mm512_shuffle_epi8(lut, hi);
    let plo = _mm512_unpacklo_epi8(hhi, hlo);
    let phi = _mm512_unpackhi_epi8(hhi, hlo);
    let pr = _mm512_cmpgt_epi8_mask(raw, _mm512_set1_epi8(0x1f))
        & _mm512_cmpgt_epi8_mask(_mm512_set1_epi8(0x7f), raw);
    let asc = _mm512_mask_blend_epi8(pr, _mm512_set1_epi8(b'.' as i8), raw);
    for r in 0..4 {
        let d = dst.add(r * row_len);
        let p = if fast_off {
            std::ptr::write_unaligned(d as *mut u64, o8[r]);
            *d.add(8) = b':';
            *d.add(9) = b' ';
            d.add(10)
        } else {
            d.add(write_prefix(d, off.wrapping_add(r as u64 * 16), core))
        };
        for cb in &layout.consts {
            store_block(p, cb);
        }
        let plo_r = match r {
            0 => _mm512_castsi512_si128(plo),
            1 => _mm512_extracti32x4_epi32(plo, 1),
            2 => _mm512_extracti32x4_epi32(plo, 2),
            _ => _mm512_extracti32x4_epi32(plo, 3),
        };
        let phi_r = match r {
            0 => _mm512_castsi512_si128(phi),
            1 => _mm512_extracti32x4_epi32(phi, 1),
            2 => _mm512_extracti32x4_epi32(phi, 2),
            _ => _mm512_extracti32x4_epi32(phi, 3),
        };
        if layout.all_fit && layout.windows.len() == 4 {
            for q in 0..4 {
                let off_q = layout.windows[q].off;
                let half = if q < 2 { plo_r } else { phi_r };
                let v = _mm_or_si128(
                    _mm_shuffle_epi8(half, _mm_loadu_si128(idx[q].as_ptr() as *const __m128i)),
                    _mm_loadu_si128(sp[q].as_ptr() as *const __m128i));
                _mm_storeu_si128(p.add(off_q) as *mut __m128i, v);
            }
        } else {
            for q in 0..4 {
                let off_q = layout.windows[q].off;
                let half = if q < 2 { plo_r } else { phi_r };
                let v = _mm_or_si128(
                    _mm_shuffle_epi8(half, _mm_loadu_si128(idx[q].as_ptr() as *const __m128i)),
                    _mm_loadu_si128(sp[q].as_ptr() as *const __m128i));
                if off_q + 16 <= layout.emitted {
                    _mm_storeu_si128(p.add(off_q) as *mut __m128i, v);
                } else {
                    let mut tmp = [0u8; 16];
                    _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, v);
                    std::ptr::copy_nonoverlapping(tmp.as_ptr(), p.add(off_q), layout.emitted - off_q);
                }
            }
        }
        let asc_r = match r {
            0 => _mm512_castsi512_si128(asc),
            1 => _mm512_extracti32x4_epi32(asc, 1),
            2 => _mm512_extracti32x4_epi32(asc, 2),
            _ => _mm512_extracti32x4_epi32(asc, 3),
        };
        if layout.ascii_run >= 16 {
            _mm_storeu_si128(p.add(layout.ascii_off) as *mut __m128i, asc_r);
        }
    }
    row_len * 4
}}

fn fast_offsets_ok(core: &RowCore, off: u64) -> bool {
    core.opts.border == BorderStyle::None
        && !core.opts.no_position
        && !core.opts.offset_dec
        && off < 0xFFFF_FFF0
}

#[cfg(target_arch = "x86_64")]
unsafe fn format_two_rows(dst: *mut u8, src: *const u8, off: u64,
                          core: &RowCore, layout: &RowLayout, row_len: usize) -> usize { unsafe {
    let lut = if core.opts.uppercase { HEX_UPPER } else { HEX_LOWER };
    let fast_off = fast_offsets_ok(core, off);
    if fast_off {
        if let Some(k) = &layout.fast {
            let o8 = hex_offsets_4(off, lut);
            return format_pair_fast(dst, src, [o8[0], o8[1]], *k, lut, row_len);
        }
    }
    let (idx, sp) = pair_masks(layout);
    let o8 = if fast_off {
        hex_offsets_4(off, lut)
    } else { [0; 4] };
    format_pair(dst, src, off, [o8[0], o8[1]], fast_off, &idx, &sp, core, layout, row_len)
}}

macro_rules! hex_offsets4 {
    ($off:expr, $lutp:expr) => {{
        let base = _mm_set1_epi32($off as i32);
        let delta = _mm_setr_epi32(0, 16, 32, 48);
        let x = _mm_add_epi32(base, delta);
        let x = _mm_shuffle_epi8(x, _mm_setr_epi8(3,2,1,0, 7,6,5,4, 11,10,9,8, 15,14,13,12));
        let m = _mm_set1_epi8(0x0f);
        let lo = _mm_and_si128(x, m);
        let hi = _mm_and_si128(_mm_srli_epi16(x, 4), m);
        let lutv = _mm_loadu_si128($lutp as *const __m128i);
        let hlo = _mm_shuffle_epi8(lutv, lo);
        let hhi = _mm_shuffle_epi8(lutv, hi);
        let plo = _mm_unpacklo_epi8(hhi, hlo);
        let phi = _mm_unpackhi_epi8(hhi, hlo);
        [_mm_cvtsi128_si64(plo) as u64,
         _mm_cvtsi128_si64(_mm_srli_si128(plo, 8)) as u64,
         _mm_cvtsi128_si64(phi) as u64,
         _mm_cvtsi128_si64(_mm_srli_si128(phi, 8)) as u64]
    }};
}

static OCT_IDX: [[i8; 16]; 4] = [
    [-1,0,-1,-1,-1,1,-1,-1,-1,2,-1,-1,-1,3,-1,-1],
    [-1,4,-1,-1,-1,5,-1,-1,-1,6,-1,-1,-1,7,-1,-1],
    [-1,8,-1,-1,-1,9,-1,-1,-1,10,-1,-1,-1,11,-1,-1],
    [-1,12,-1,-1,-1,13,-1,-1,-1,14,-1,-1,-1,15,-1,-1],
];
static OCT_IDX1: [[i8; 16]; 4] = [
    [-1,-1,0,-1,-1,-1,1,-1,-1,-1,2,-1,-1,-1,3,-1],
    [-1,-1,4,-1,-1,-1,5,-1,-1,-1,6,-1,-1,-1,7,-1],
    [-1,-1,8,-1,-1,-1,9,-1,-1,-1,10,-1,-1,-1,11,-1],
    [-1,-1,12,-1,-1,-1,13,-1,-1,-1,14,-1,-1,-1,15,-1],
];
static OCT_IDX2: [[i8; 16]; 4] = [
    [-1,-1,-1,0,-1,-1,-1,1,-1,-1,-1,2,-1,-1,-1,3],
    [-1,-1,-1,4,-1,-1,-1,5,-1,-1,-1,6,-1,-1,-1,7],
    [-1,-1,-1,8,-1,-1,-1,9,-1,-1,-1,10,-1,-1,-1,11],
    [-1,-1,-1,12,-1,-1,-1,13,-1,-1,-1,14,-1,-1,-1,15],
];
static OCT_SP: [u8; 16] = [32,0,0,0,32,0,0,0,32,0,0,0,32,0,0,0];
static OCT_MASK_D0: [u8; 16] = [3,7,3,7,3,7,3,7,3,7,3,7,3,7,3,7];

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3,sse4.1")]
unsafe fn format_row_octal(dst: *mut u8, src: *const u8, o8: u64) { unsafe {
    let raw = _mm_loadu_si128(src as *const __m128i);
    let m7 = _mm_set1_epi8(0x07);
    let m0d = _mm_loadu_si128(OCT_MASK_D0.as_ptr() as *const __m128i);
    let d0 = _mm_and_si128(_mm_srli_epi16(raw, 6), m0d);
    let d1 = _mm_and_si128(_mm_srli_epi16(raw, 3), m7);
    let d2 = _mm_and_si128(raw, m7);
    let lut = _mm_loadu_si128(HEX_LOWER.as_ptr() as *const __m128i);
    let c0 = _mm_shuffle_epi8(lut, d0);
    let c1 = _mm_shuffle_epi8(lut, d1);
    let c2 = _mm_shuffle_epi8(lut, d2);
    std::ptr::write_unaligned(dst as *mut u64, o8);
    *(dst.add(8) as *mut u16) = 0x203a;
    let p = dst.add(10);
    let sp = _mm_loadu_si128(OCT_SP.as_ptr() as *const __m128i);
    for w in 0..4 {
        let idx0 = _mm_loadu_si128(OCT_IDX[w].as_ptr() as *const __m128i);
        let idx1 = _mm_loadu_si128(OCT_IDX1[w].as_ptr() as *const __m128i);
        let idx2 = _mm_loadu_si128(OCT_IDX2[w].as_ptr() as *const __m128i);
        let v = _mm_or_si128(_mm_or_si128(
            _mm_or_si128(_mm_shuffle_epi8(c0, idx0), _mm_shuffle_epi8(c1, idx1)),
            _mm_shuffle_epi8(c2, idx2)),
            sp);
        _mm_storeu_si128(p.add(w * 16) as *mut __m128i, v);
    }
    *p.add(64) = b'\n';
}}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3,sse4.1")]
unsafe fn format_octal_run(dst: *mut u8, src: *const u8, off: u64, rows: usize,
                           upper: bool, row_len: usize) { unsafe {
    let mut i = 0;
    let lut = if upper { HEX_UPPER.as_ptr() } else { HEX_LOWER.as_ptr() };
    while i + 3 < rows {
        let o8 = hex_offsets4!(off.wrapping_add((i * 16) as u64), lut);
        format_row_octal(dst.add(i * row_len), src.add(i * 16), o8[0]);
        format_row_octal(dst.add((i + 1) * row_len), src.add((i + 1) * 16), o8[1]);
        format_row_octal(dst.add((i + 2) * row_len), src.add((i + 2) * 16), o8[2]);
        format_row_octal(dst.add((i + 3) * row_len), src.add((i + 3) * 16), o8[3]);
        i += 4;
    }
    while i < rows {
        let o8 = hex_offsets4!(off.wrapping_add((i * 16) as u64), lut);
        format_row_octal(dst.add(i * row_len), src.add(i * 16), o8[0]);
        i += 1;
    }
}}

macro_rules! fast_pair_min_body {
    ($dst:expr, $src:expr, $o0:expr, $o1:expr, $lutp:expr, $has_ascii:expr, $row_len:expr) => {{
        let raw = _mm256_loadu_si256($src as *const __m256i);
        let m0f = _mm256_set1_epi8(0x0f);
        let lo  = _mm256_and_si256(raw, m0f);
        let hi  = _mm256_and_si256(_mm256_srli_epi16(raw, 4), m0f);
        let lutv = _mm256_broadcastsi128_si256(_mm_loadu_si128($lutp as *const __m128i));
        let hlo = _mm256_shuffle_epi8(lutv, lo);
        let hhi = _mm256_shuffle_epi8(lutv, hi);
        let plo = _mm256_unpacklo_epi8(hhi, hlo);
        let phi = _mm256_unpackhi_epi8(hhi, hlo);
        let asc = if $has_ascii {
            let pr = _mm256_and_si256(
                _mm256_cmpgt_epi8(raw, _mm256_set1_epi8(0x1f)),
                _mm256_cmpgt_epi8(_mm256_set1_epi8(0x7f), raw));
            _mm256_blendv_epi8(_mm256_set1_epi8(b'.' as i8), raw, pr)
        } else {
            _mm256_setzero_si256()
        };
        let d0 = $dst;
        std::ptr::write_unaligned(d0 as *mut u64, $o0);
        *d0.add(8) = b' ';
        let d1 = d0.add($row_len);
        std::ptr::write_unaligned(d1 as *mut u64, $o1);
        *d1.add(8) = b' ';
        let p0 = d0.add(9);
        let plo0 = _mm256_castsi256_si128(plo);
        let phi0 = _mm256_castsi256_si128(phi);
        let h0 = _mm256_inserti128_si256(_mm256_castsi128_si256(plo0), phi0, 1);
        _mm256_storeu_si256(p0 as *mut __m256i, h0);
        if $has_ascii {
            let asc0 = _mm256_castsi256_si128(asc);
            *p0.add(32) = b' ';
            _mm_storeu_si128(p0.add(33) as *mut __m128i, asc0);
            *p0.add(49) = b'\n';
        } else {
            *p0.add(32) = b'\n';
        }
        let p1 = d1.add(9);
        let plo1 = _mm256_extracti128_si256(plo, 1);
        let phi1 = _mm256_extracti128_si256(phi, 1);
        let h1 = _mm256_inserti128_si256(_mm256_castsi128_si256(plo1), phi1, 1);
        _mm256_storeu_si256(p1 as *mut __m256i, h1);
        if $has_ascii {
            let asc1 = _mm256_extracti128_si256(asc, 1);
            *p1.add(32) = b' ';
            _mm_storeu_si128(p1.add(33) as *mut __m128i, asc1);
            *p1.add(49) = b'\n';
        } else {
            *p1.add(32) = b'\n';
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,ssse3,sse4.1")]
unsafe fn format_pairs_min_run(dst: *mut u8, src: *const u8, off: u64, pairs: usize,
                               lut: &[u8; 16], has_ascii: bool, row_len: usize) { unsafe {
    let lutp = lut.as_ptr();
    let mut i = 0;
    while i + 1 < pairs {
        let o8 = hex_offsets4!(off.wrapping_add((i * 32) as u64), lutp);
        fast_pair_min_body!(dst.add(i * row_len * 2), src.add(i * 32), o8[0], o8[1], lutp, has_ascii, row_len);
        fast_pair_min_body!(dst.add((i + 1) * row_len * 2), src.add((i + 1) * 32), o8[2], o8[3], lutp, has_ascii, row_len);
        i += 2;
    }
    if i < pairs {
        let o8 = hex_offsets4!(off.wrapping_add((i * 32) as u64), lutp);
        fast_pair_min_body!(dst.add(i * row_len * 2), src.add(i * 32), o8[0], o8[1], lutp, has_ascii, row_len);
    }
}}

macro_rules! fast_pair_head {
    ($src:expr, $lutp:expr) => {{
        let raw = _mm256_loadu_si256($src as *const __m256i);
        let m0f = _mm256_set1_epi8(0x0f);
        let lo  = _mm256_and_si256(raw, m0f);
        let hi  = _mm256_and_si256(_mm256_srli_epi16(raw, 4), m0f);
        let lutv = _mm256_broadcastsi128_si256(_mm_loadu_si128($lutp as *const __m128i));
        let hlo = _mm256_shuffle_epi8(lutv, lo);
        let hhi = _mm256_shuffle_epi8(lutv, hi);
        let plo = _mm256_unpacklo_epi8(hhi, hlo);
        let phi = _mm256_unpackhi_epi8(hhi, hlo);
        (raw, plo, phi)
    }};
}

macro_rules! fast_pair_ascii {
    ($dst:expr, $src:expr, $o0:expr, $o1:expr, $lutp:expr, $k:expr, $row_len:expr) => {{
        let (raw, plo, phi) = fast_pair_head!($src, $lutp);
        let pr = _mm256_and_si256(
            _mm256_cmpgt_epi8(raw, _mm256_set1_epi8(0x1f)),
            _mm256_cmpgt_epi8(_mm256_set1_epi8(0x7f), raw));
        let asc = _mm256_blendv_epi8(_mm256_set1_epi8(b'.' as i8), raw, pr);
        let d0 = $dst;
        std::ptr::write_unaligned(d0 as *mut u64, $o0);
        *(d0.add(8) as *mut u16) = 0x203a;
        let d1 = d0.add($row_len);
        std::ptr::write_unaligned(d1 as *mut u64, $o1);
        *(d1.add(8) as *mut u16) = 0x203a;
        let p0 = d0.add(10);
        let plo0 = _mm256_castsi256_si128(plo);
        let phi0 = _mm256_castsi256_si128(phi);
        let asc0 = _mm256_castsi256_si128(asc);
        let e0 = _mm_or_si128(
            _mm_shuffle_epi8(plo0, _mm_loadu_si128($k.idx_a[0].as_ptr() as *const __m128i)),
            _mm_loadu_si128($k.sp[0].as_ptr() as *const __m128i));
        let e1 = _mm_or_si128(_mm_or_si128(
            _mm_shuffle_epi8(plo0, _mm_loadu_si128($k.idx_a[1].as_ptr() as *const __m128i)),
            _mm_shuffle_epi8(phi0, _mm_loadu_si128($k.idx_b[1].as_ptr() as *const __m128i))),
            _mm_loadu_si128($k.sp[1].as_ptr() as *const __m128i));
        let fh = _mm_or_si128(_mm_or_si128(
            _mm_shuffle_epi8(plo0, _mm_loadu_si128($k.idx_a[2].as_ptr() as *const __m128i)),
            _mm_shuffle_epi8(phi0, _mm_loadu_si128($k.idx_b[2].as_ptr() as *const __m128i))),
            _mm_loadu_si128($k.sp[2].as_ptr() as *const __m128i));
        let ft = _mm_or_si128(
            _mm_shuffle_epi8(asc0, _mm_loadu_si128($k.w3_idx.as_ptr() as *const __m128i)),
            _mm_loadu_si128($k.w3_sp.as_ptr() as *const __m128i));
        let w0 = _mm256_inserti128_si256(_mm256_castsi128_si256(e0), e1, 1);
        let w1 = _mm256_inserti128_si256(_mm256_castsi128_si256(fh), ft, 1);
        _mm256_storeu_si256(p0 as *mut __m256i, w0);
        _mm256_storeu_si256(p0.add(32) as *mut __m256i, w1);
        let tu = _mm_or_si128(
            _mm_shuffle_epi8(asc0, _mm_loadu_si128($k.tu_idx.as_ptr() as *const __m128i)),
            _mm_loadu_si128($k.tu_sp.as_ptr() as *const __m128i));
        *(p0.add(64) as *mut u32) = _mm_cvtsi128_si64(tu) as u32;
        *p0.add(68) = $k.nl;
        let p1 = d1.add(10);
        let plo1 = _mm256_extracti128_si256(plo, 1);
        let phi1 = _mm256_extracti128_si256(phi, 1);
        let asc1 = _mm256_extracti128_si256(asc, 1);
        let g0 = _mm_or_si128(
            _mm_shuffle_epi8(plo1, _mm_loadu_si128($k.idx_a[0].as_ptr() as *const __m128i)),
            _mm_loadu_si128($k.sp[0].as_ptr() as *const __m128i));
        let g1 = _mm_or_si128(_mm_or_si128(
            _mm_shuffle_epi8(plo1, _mm_loadu_si128($k.idx_a[1].as_ptr() as *const __m128i)),
            _mm_shuffle_epi8(phi1, _mm_loadu_si128($k.idx_b[1].as_ptr() as *const __m128i))),
            _mm_loadu_si128($k.sp[1].as_ptr() as *const __m128i));
        let gh = _mm_or_si128(_mm_or_si128(
            _mm_shuffle_epi8(plo1, _mm_loadu_si128($k.idx_a[2].as_ptr() as *const __m128i)),
            _mm_shuffle_epi8(phi1, _mm_loadu_si128($k.idx_b[2].as_ptr() as *const __m128i))),
            _mm_loadu_si128($k.sp[2].as_ptr() as *const __m128i));
        let gt = _mm_or_si128(
            _mm_shuffle_epi8(asc1, _mm_loadu_si128($k.w3_idx.as_ptr() as *const __m128i)),
            _mm_loadu_si128($k.w3_sp.as_ptr() as *const __m128i));
        let w2 = _mm256_inserti128_si256(_mm256_castsi128_si256(g0), g1, 1);
        let w3 = _mm256_inserti128_si256(_mm256_castsi128_si256(gh), gt, 1);
        _mm256_storeu_si256(p1 as *mut __m256i, w2);
        _mm256_storeu_si256(p1.add(32) as *mut __m256i, w3);
        let gu = _mm_or_si128(
            _mm_shuffle_epi8(asc1, _mm_loadu_si128($k.tu_idx.as_ptr() as *const __m128i)),
            _mm_loadu_si128($k.tu_sp.as_ptr() as *const __m128i));
        *(p1.add(64) as *mut u32) = _mm_cvtsi128_si64(gu) as u32;
        *p1.add(68) = $k.nl;
    }};
}

macro_rules! fast_pair_na {
    ($dst:expr, $src:expr, $o0:expr, $o1:expr, $lutp:expr, $k:expr, $row_len:expr) => {{
        let (_raw, plo, phi) = fast_pair_head!($src, $lutp);
        let d0 = $dst;
        std::ptr::write_unaligned(d0 as *mut u64, $o0);
        *(d0.add(8) as *mut u16) = 0x203a;
        let d1 = d0.add($row_len);
        std::ptr::write_unaligned(d1 as *mut u64, $o1);
        *(d1.add(8) as *mut u16) = 0x203a;
        let p0 = d0.add(10);
        let plo0 = _mm256_castsi256_si128(plo);
        let phi0 = _mm256_castsi256_si128(phi);
        let e0 = _mm_or_si128(
            _mm_shuffle_epi8(plo0, _mm_loadu_si128($k.idx_a[0].as_ptr() as *const __m128i)),
            _mm_loadu_si128($k.sp[0].as_ptr() as *const __m128i));
        let e1 = _mm_or_si128(_mm_or_si128(
            _mm_shuffle_epi8(plo0, _mm_loadu_si128($k.idx_a[1].as_ptr() as *const __m128i)),
            _mm_shuffle_epi8(phi0, _mm_loadu_si128($k.idx_b[1].as_ptr() as *const __m128i))),
            _mm_loadu_si128($k.sp[1].as_ptr() as *const __m128i));
        let fh = _mm_or_si128(_mm_or_si128(
            _mm_shuffle_epi8(plo0, _mm_loadu_si128($k.idx_a[2].as_ptr() as *const __m128i)),
            _mm_shuffle_epi8(phi0, _mm_loadu_si128($k.idx_b[2].as_ptr() as *const __m128i))),
            _mm_loadu_si128($k.sp[2].as_ptr() as *const __m128i));
        let ft = _mm_loadu_si128($k.w3_sp.as_ptr() as *const __m128i);
        let w0 = _mm256_inserti128_si256(_mm256_castsi128_si256(e0), e1, 1);
        _mm256_storeu_si256(p0 as *mut __m256i, w0);
        _mm_storeu_si128(p0.add(32) as *mut __m128i, fh);
        *p0.add(48) = _mm_cvtsi128_si64(ft) as u8;
        let p1 = d1.add(10);
        let plo1 = _mm256_extracti128_si256(plo, 1);
        let phi1 = _mm256_extracti128_si256(phi, 1);
        let g0 = _mm_or_si128(
            _mm_shuffle_epi8(plo1, _mm_loadu_si128($k.idx_a[0].as_ptr() as *const __m128i)),
            _mm_loadu_si128($k.sp[0].as_ptr() as *const __m128i));
        let g1 = _mm_or_si128(_mm_or_si128(
            _mm_shuffle_epi8(plo1, _mm_loadu_si128($k.idx_a[1].as_ptr() as *const __m128i)),
            _mm_shuffle_epi8(phi1, _mm_loadu_si128($k.idx_b[1].as_ptr() as *const __m128i))),
            _mm_loadu_si128($k.sp[1].as_ptr() as *const __m128i));
        let gh = _mm_or_si128(_mm_or_si128(
            _mm_shuffle_epi8(plo1, _mm_loadu_si128($k.idx_a[2].as_ptr() as *const __m128i)),
            _mm_shuffle_epi8(phi1, _mm_loadu_si128($k.idx_b[2].as_ptr() as *const __m128i))),
            _mm_loadu_si128($k.sp[2].as_ptr() as *const __m128i));
        let gt = _mm_loadu_si128($k.w3_sp.as_ptr() as *const __m128i);
        let w2 = _mm256_inserti128_si256(_mm256_castsi128_si256(g0), g1, 1);
        _mm256_storeu_si256(p1 as *mut __m256i, w2);
        _mm_storeu_si128(p1.add(32) as *mut __m128i, gh);
        *p1.add(48) = _mm_cvtsi128_si64(gt) as u8;
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,ssse3,sse4.1")]
unsafe fn format_pair_fast(dst: *mut u8, src: *const u8, o8: [u64; 2],
                           k: CanonKernel, lut: &[u8; 16], row_len: usize) -> usize { unsafe {
    let lutp = lut.as_ptr();
    if k.has_ascii {
        fast_pair_ascii!(dst, src, o8[0], o8[1], lutp, &k, row_len);
    } else {
        fast_pair_na!(dst, src, o8[0], o8[1], lutp, &k, row_len);
    }
    row_len * 2
}}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,ssse3,sse4.1")]
unsafe fn format_pairs_fast_run(dst: *mut u8, src: *const u8, off: u64, pairs: usize,
                                lut: &[u8; 16], k: CanonKernel, row_len: usize) { unsafe {
    let lutp = lut.as_ptr();
    let mut i = 0;
    if k.has_ascii {
        while i + 1 < pairs {
            let o8 = hex_offsets4!(off.wrapping_add((i * 32) as u64), lutp);
            fast_pair_ascii!(dst.add(i * row_len * 2), src.add(i * 32), o8[0], o8[1], lutp, &k, row_len);
            fast_pair_ascii!(dst.add((i + 1) * row_len * 2), src.add((i + 1) * 32), o8[2], o8[3], lutp, &k, row_len);
            i += 2;
        }
        if i < pairs {
            let o8 = hex_offsets4!(off.wrapping_add((i * 32) as u64), lutp);
            fast_pair_ascii!(dst.add(i * row_len * 2), src.add(i * 32), o8[0], o8[1], lutp, &k, row_len);
        }
    } else {
        while i + 1 < pairs {
            let o8 = hex_offsets4!(off.wrapping_add((i * 32) as u64), lutp);
            fast_pair_na!(dst.add(i * row_len * 2), src.add(i * 32), o8[0], o8[1], lutp, &k, row_len);
            fast_pair_na!(dst.add((i + 1) * row_len * 2), src.add((i + 1) * 32), o8[2], o8[3], lutp, &k, row_len);
            i += 2;
        }
        if i < pairs {
            let o8 = hex_offsets4!(off.wrapping_add((i * 32) as u64), lutp);
            fast_pair_na!(dst.add(i * row_len * 2), src.add(i * 32), o8[0], o8[1], lutp, &k, row_len);
        }
    }
}}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,ssse3,sse4.1")]
unsafe fn format_pairs_run(dst: *mut u8, src: *const u8, off: u64, pairs: usize,
                           idx: &[[i8; 16]; 4], sp: &[[u8; 16]; 4], fast: bool,
                           lut: &[u8; 16],
                           core: &RowCore, layout: &RowLayout, row_len: usize) { unsafe {
    for k in 0..pairs {
        let o8 = if fast {
            hex_offsets_4(off.wrapping_add((k * 32) as u64), lut)
        } else { [0; 4] };
        format_pair(dst.add(k * row_len * 2), src.add(k * 32),
                    off.wrapping_add((k * 32) as u64), [o8[0], o8[1]], fast,
                    idx, sp, core, layout, row_len);
    }
}}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn format_fours_run(dst: *mut u8, src: *const u8, off: u64, quads: usize,
                           idx: &[[i8; 16]; 4], sp: &[[u8; 16]; 4], fast: bool,
                           lut: &[u8; 16],
                           core: &RowCore, layout: &RowLayout, row_len: usize) { unsafe {
    for k in 0..quads {
        let o8 = if fast {
            hex_offsets_4(off.wrapping_add((k * 64) as u64), lut)
        } else { [0; 4] };
        format_four_rows(dst.add(k * row_len * 4), src.add(k * 64),
                         off.wrapping_add((k * 64) as u64), o8, fast,
                         idx, sp, core, layout, row_len);
    }
}}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn format_row(dst: *mut u8, src: *const u8, off: u64, n: usize,
                     core: &RowCore, layout: &RowLayout) -> usize {
    let _ = (src, n);
    let mut v = Vec::new();
    let slice = std::slice::from_raw_parts(src, n);
    format_row_generic(&mut v, slice, off, &core.opts);
    let plen = write_prefix(dst, off, core);
    // generic includes prefix; just copy whole row
    let _ = plen;
    std::ptr::copy_nonoverlapping(v.as_ptr(), dst, v.len());
    let _ = layout;
    v.len()
}
#[cfg(not(target_arch = "x86_64"))]
unsafe fn format_two_rows(dst: *mut u8, src: *const u8, off: u64,
                          core: &RowCore, layout: &RowLayout, row_len: usize) -> usize {
    format_row(dst, src, off, 16, core, layout);
    format_row(dst.add(row_len), src.add(16), off.wrapping_add(16), 16, core, layout);
    row_len * 2
}
#[cfg(not(target_arch = "x86_64"))]
unsafe fn format_pairs_min_run(dst: *mut u8, src: *const u8, off: u64, pairs: usize,
                               lut: &[u8; 16], has_ascii: bool, row_len: usize) {
    let _ = (dst, src, off, pairs, lut, has_ascii, row_len);
}
#[cfg(not(target_arch = "x86_64"))]
unsafe fn format_pairs_fast_run(dst: *mut u8, src: *const u8, off: u64, pairs: usize,
                                lut: &[u8; 16], k: CanonKernel, row_len: usize) {
    let _ = (dst, src, off, pairs, lut, k, row_len);
}
#[cfg(not(target_arch = "x86_64"))]
unsafe fn format_pairs_run(dst: *mut u8, src: *const u8, off: u64, pairs: usize,
                           idx: &[[i8; 16]; 4], sp: &[[u8; 16]; 4], fast: bool,
                           lut: &[u8; 16],
                           core: &RowCore, layout: &RowLayout, row_len: usize) {
    let _ = (dst, src, off, pairs, idx, sp, fast, lut, core, layout, row_len);
}
#[cfg(not(target_arch = "x86_64"))]
unsafe fn format_fours_run(dst: *mut u8, src: *const u8, off: u64, quads: usize,
                           idx: &[[i8; 16]; 4], sp: &[[u8; 16]; 4], fast: bool,
                           lut: &[u8; 16],
                           core: &RowCore, layout: &RowLayout, row_len: usize) {
    let _ = (dst, src, off, quads, idx, sp, fast, lut, core, layout, row_len);
}
#[cfg(not(target_arch = "x86_64"))]
unsafe fn format_octal_run(dst: *mut u8, src: *const u8, off: u64, rows: usize,
                           upper: bool, row_len: usize) {
    let _ = (dst, src, off, rows, upper, row_len);
}
#[cfg(not(target_arch = "x86_64"))]
unsafe fn format_pair(dst: *mut u8, src: *const u8, off: u64, o8: [u64; 2], fast_off: bool,
                      idx: &[[i8; 16]; 4], sp: &[[u8; 16]; 4],
                      core: &RowCore, layout: &RowLayout, row_len: usize) -> usize {
    let _ = (dst, src, off, o8, fast_off, idx, sp, core, layout, row_len);
    0
}
#[cfg(not(target_arch = "x86_64"))]
unsafe fn hex_offsets_4(_off: u64, _lut: &[u8; 16]) -> [u64; 4] { [0; 4] }
#[cfg(not(target_arch = "x86_64"))]
unsafe fn plain_blocks(dst: *mut u8, src: *const u8, blocks: usize, upper: bool) {
    let lut = if upper { HEX_UPPER } else { HEX_LOWER };
    for i in 0..blocks {
        encode_hex16_portable(
            unsafe { std::slice::from_raw_parts(src.add(i * 16), 16) },
            lut,
            unsafe { std::slice::from_raw_parts_mut(dst.add(i * 32), 32) },
        );
    }
}

fn _write_hex_group(dst: &mut Vec<u8>, src: &[u8], group: usize, endian: Endian,
                   upper: bool, sep: u8) {
    let hex = if upper { HEX_UPPER } else { HEX_LOWER };
    let len = group.min(src.len());
    dst.push(sep);
    let iter: Box<dyn Iterator<Item=u8>> = if endian == Endian::Little {
        Box::new(src[..len].iter().copied().rev())
    } else {
        Box::new(src[..len].iter().copied())
    };
    for b in iter { dst.push(hex[(b>>4) as usize]); dst.push(hex[(b&0xf) as usize]); }
    // pad if partial
    for _ in len..group { dst.push(b' '); dst.push(b' '); }
}

fn ascii_byte(b: u8, table: CharTable, buf: &mut Vec<u8>) {
    match table {
        CharTable::Ascii => {
            buf.push(if b >= 0x20 && b <= 0x7e { b } else { b'.' });
        }
        CharTable::Default => {
            match b {
                0x00 => buf.extend_from_slice("⋄".as_bytes()),
                0x20 => buf.push(b' '),
                0x21..=0x7e => buf.push(b),
                _ => buf.extend_from_slice("•".as_bytes()),
            }
        }
        CharTable::Braille => {
            let encoded = braille_for_byte(b);
            buf.extend_from_slice(&encoded);
        }
        CharTable::Cp437 => {
            let c = CP437[b as usize];
            let mut tmp = [0u8; 4];
            buf.extend_from_slice(c.encode_utf8(&mut tmp).as_bytes());
        }
        CharTable::Ebcdic => {
            let c = EBCDIC_TO_ASCII[b as usize];
            buf.push(if c >= 0x20 && c <= 0x7e { c } else { b'.' });
        }
    }
}

fn hex_width(opts: &Options) -> usize {
    let bpr = opts.width;
    let group = opts.group;
    match opts.mode {
        DisplayMode::Canonical => {
            if opts.minimal { return bpr * 2; }
            let mut w = 0;
            for i in 0..bpr {
                if i > 0 {
                    if i % (bpr / 2) == 0 { w += 1; }
                    else if group > 1 && i % group == 0 { w += 1; }
                }
                w += 2; // hex digits
                if i < bpr - 1 { w += 1; } // space between bytes
            }
            w
        }
        DisplayMode::OneByteHex => bpr * 4,
        DisplayMode::OneByteOctal => bpr * 4,
        DisplayMode::OneByteDecimal => bpr * 4,
        DisplayMode::TwoByteHex => {
            let g = group.max(2);
            (bpr / g) * (g * 2 + 1)
        }
        DisplayMode::TwoByteOctal => {
            let g = group.max(2);
            (bpr / g) * (g * 3 + 1)
        }
        DisplayMode::TwoByteDecimal => {
            let g = group.max(2);
            (bpr / g) * 6
        }
        DisplayMode::OneByteChar => bpr * 4,
        DisplayMode::Binary => 8 * 9,
        _ => bpr * 3,
    }
}

fn ascii_width(opts: &Options) -> usize {
    opts.width
}

fn format_row_generic(
    dst:         &mut Vec<u8>,
    src:         &[u8],
    display_off: u64,
    opts:        &Options,
) {
    let hex = if opts.uppercase { HEX_UPPER } else { HEX_LOWER };
    let width  = opts.width;
    let group  = opts.group;
    let n      = src.len();

    if opts.minimal && opts.mode == DisplayMode::Canonical {
        if !opts.no_position {
            let mut tmp = [0u8; 20];
            let olen = write_offset(&mut tmp, display_off, opts.offset_dec, opts.uppercase);
            dst.extend_from_slice(&tmp[..olen]);
            dst.push(b' ');
        }
        for i in 0..n {
            let b = src[i];
            dst.push(hex[(b >> 4) as usize]);
            dst.push(hex[(b & 0xf) as usize]);
        }
        if !opts.no_ascii {
            dst.push(b' ');
            for i in 0..n {
                let b = src[i];
                dst.push(if b >= 0x20 && b <= 0x7e { b } else { b'.' });
            }
        }
        dst.push(b'\n');
        return;
    }

    let left_bar  = match opts.border { BorderStyle::None=>"", BorderStyle::Ascii=>"|", BorderStyle::Unicode=>"│" };
    let mid_bar   = left_bar;
    let right_bar = left_bar;

    // Offset column
    if !opts.no_position {
        if opts.border != BorderStyle::None { dst.extend_from_slice(left_bar.as_bytes()); }
        let mut tmp = [0u8; 20];
        let olen = write_offset(&mut tmp, display_off, opts.offset_dec, opts.uppercase);
        dst.extend_from_slice(&tmp[..olen]);
        dst.push(b':');
        if opts.border != BorderStyle::None {
            let pos_w = if opts.offset_dec { 20 } else { offset_len(u64::MAX) } + 1;
            let pad = pos_w.saturating_sub(olen + 1);
            for _ in 0..pad { dst.push(b' '); }
        } else {
            dst.push(b' ');
        }
    }

    // Hex section
    if opts.border != BorderStyle::None {
        if opts.no_position { dst.extend_from_slice(left_bar.as_bytes()); }
        else                { dst.extend_from_slice(mid_bar.as_bytes()); }
    }
    let hex_start = dst.len();

    match opts.mode {
        DisplayMode::Canonical => {
            for i in 0..width {
                if i > 0 {
                    if i % (width / 2) == 0 { dst.push(b' '); }
                    else if opts.group > 1 && i % opts.group == 0 { dst.push(b' '); }
                }
                if i < n {
                    let b = if opts.endian == Endian::Little && group > 1 {
                        let grp_start = (i / group) * group;
                        let within    = i % group;
                        let rev_idx   = grp_start + (group - 1 - within);
                        if rev_idx < n { src[rev_idx] } else { 0 }
                    } else { src[i] };
                    dst.push(hex[(b >> 4) as usize]);
                    dst.push(hex[(b & 0xf) as usize]);
                    if i < width - 1 || (!opts.no_ascii && opts.border == BorderStyle::None) { dst.push(b' '); }
                } else {
                    dst.push(b' '); dst.push(b' '); 
                    if i < width - 1 || (!opts.no_ascii && opts.border == BorderStyle::None) { dst.push(b' '); }
                }
            }
        }
        DisplayMode::OneByteHex => {
            for i in 0..width {
                if i < n { dst.push(b' '); dst.push(b' ');
                    dst.push(hex[(src[i]>>4) as usize]);
                    dst.push(hex[(src[i]&0xf) as usize]);
                } else { dst.extend_from_slice(b"    "); }
            }
        }
        DisplayMode::TwoByteHex => {
            let g = opts.group.max(2);
            let pairs = width / 2;
            for i in 0..pairs {
                let bi = i * g;
                if bi < n {
                    let v = read_le_u64(&src[bi..], g.min(n-bi), opts.endian);
                    // pad to g*2 hex digits
                    dst.push(b' ');
                    for k in (0..g*2).rev() {
                        dst.push(hex[((v >> (k*4)) & 0xf) as usize]);
                    }
                } else {
                    for _ in 0..g*2+1 { dst.push(b' '); }
                }
            }
        }
        DisplayMode::OneByteOctal => {
            for i in 0..width {
                dst.push(b' ');
                if i < n { let b=src[i]; dst.push(b'0'+(b>>6)); dst.push(b'0'+((b>>3)&7)); dst.push(b'0'+(b&7)); }
                else { dst.extend_from_slice(b"   "); }
            }
        }
        DisplayMode::TwoByteOctal => {
            let g = opts.group.max(2);
            for i in 0..(width/2) {
                let bi = i*g;
                dst.push(b' ');
                if bi < n {
                    let v = read_le_u64(&src[bi..], g.min(n-bi), opts.endian);
                    // 6 octal digits for u16
                    let digits = g * 3;
                    for k in (0..digits).rev() {
                        dst.push(b'0' + ((v >> (k*3)) & 7) as u8);
                    }
                } else {
                    for _ in 0..g*3 { dst.push(b' '); }
                }
            }
        }
        DisplayMode::OneByteDecimal => {
            for i in 0..width {
                dst.push(b' ');
                if i < n {
                    let b = src[i];
                    dst.push(b'0' + b/100);
                    dst.push(b'0' + (b/10)%10);
                    dst.push(b'0' + b%10);
                } else { dst.extend_from_slice(b"   "); }
            }
        }
        DisplayMode::TwoByteDecimal => {
            let g = opts.group.max(2);
            for i in 0..(width/2) {
                let bi = i*g;
                if bi < n {
                    let v = read_le_u64(&src[bi..], g.min(n-bi), opts.endian);
                    dst.push(b' ');
                    // max 5 decimal digits for u16
                    let s = format!("{:05}", v as u16);
                    dst.extend_from_slice(s.as_bytes());
                } else { dst.extend_from_slice(b"      "); }
            }
        }
        DisplayMode::OneByteChar => {
            for i in 0..width {
                if i < n {
                    let b = src[i];
                    match b {
                        0x00 => dst.extend_from_slice(b"  \0"),
                        0x07 => dst.extend_from_slice(b"  \x07"),
                        0x08 => dst.extend_from_slice(b"  \x08"),
                        0x09 => dst.extend_from_slice(b"  \t"),
                        0x0a => dst.extend_from_slice(b"  \n"),
                        0x0b => dst.extend_from_slice(b"  \x0b"),
                        0x0c => dst.extend_from_slice(b"  \x0c"),
                        0x0d => dst.extend_from_slice(b"  \r"),
                        0x20..=0x7e => { dst.push(b' '); dst.push(b' '); dst.push(b' '); dst.push(b); }
                        _ => { dst.push(b' '); dst.push(b'0'+(b>>6)); dst.push(b'0'+((b>>3)&7)); dst.push(b'0'+(b&7)); }
                    }
                } else { dst.extend_from_slice(b"    "); }
            }
        }
        DisplayMode::Binary => {
            for i in 0..8usize {
                dst.push(b' ');
                if i < n {
                    let b = src[i];
                    for bit in (0..8).rev() { dst.push(b'0' + ((b >> bit) & 1)); }
                } else { dst.extend_from_slice(b"        "); }
            }
        }
        _ => {} // Plain/CInclude/Reverse don't use this path
    }

    // Pad Hex section to match border
    if opts.border != BorderStyle::None {
        let hex_len = dst.len() - hex_start;
        let pad = hex_width(opts).saturating_sub(hex_len);
        for _ in 0..pad { dst.push(b' '); }
    }

    // ASCII panel
    if !opts.no_ascii && !matches!(opts.mode,
        DisplayMode::Binary | DisplayMode::OneByteOctal | DisplayMode::TwoByteOctal |
        DisplayMode::OneByteDecimal | DisplayMode::TwoByteDecimal |
        DisplayMode::OneByteChar | DisplayMode::OneByteHex | DisplayMode::TwoByteHex)
    {
        if opts.border != BorderStyle::None {
            dst.push(b' ');
            dst.extend_from_slice(mid_bar.as_bytes());
        } else {
            dst.push(b' ');
            dst.push(b'|');
        }
        let ascii_start = dst.len();
        for i in 0..n { ascii_byte(src[i], opts.table, dst); }
        if opts.border != BorderStyle::None {
            let current_len = dst.len() - ascii_start;
            let pad = ascii_width(opts).saturating_sub(current_len);
            for _ in 0..pad { dst.push(b' '); }
            dst.push(b' ');
            dst.extend_from_slice(right_bar.as_bytes());
        } else {
            dst.push(b'|');
        }
    } else {
        if opts.border != BorderStyle::None {
            dst.push(b' ');
            dst.extend_from_slice(right_bar.as_bytes());
        }
    }

    dst.push(b'\n');
}

fn read_le_u64(src: &[u8], len: usize, endian: Endian) -> u64 {
    let mut v = 0u64;
    match endian {
        Endian::Big => {
            for i in 0..len { v = (v << 8) | src[i] as u64; }
        }
        Endian::Little => {
            for i in 0..len { v |= (src[i] as u64) << (i*8); }
        }
    }
    v
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,ssse3,sse4.1")]
unsafe fn plain_blocks(dst: *mut u8, src: *const u8, blocks: usize, upper: bool) { unsafe {
    let lut = if upper { HEX_UPPER } else { HEX_LOWER };
    for i in 0..blocks {
        encode_hex16_portable(
            std::slice::from_raw_parts(src.add(i * 16), 16),
            lut,
            std::slice::from_raw_parts_mut(dst.add(i * 32), 32),
        );
    }
}}

fn run_plain_mmap(opts: &Options, data: &[u8]) -> io::Result<()> {
    let blocks = data.len() / 16;
    let tail_len = data.len() % 16;
    let chunk_blocks = ((16 * 1024 * 1024) / 32).max(1);
    let buf_cap = chunk_blocks * 32 + 16;

    let (send_data, recv_data) = sync_channel::<Vec<u8>>(5);
    let (send_free, recv_free) = channel::<Vec<u8>>();
    for _ in 0..6 {
        let buf = vec![0u8; buf_cap];
        send_free.send(buf).unwrap();
    }
    let writer = thread::spawn(move || -> io::Result<()> {
        let mut zc = ZeroCopyWriter::new()?;
        let mut prev: Option<Vec<u8>> = None;
        while let Ok(chunk) = recv_data.recv() {
            zc.write_chunk(&chunk)?;
            if let Some(p) = prev.take() {
                let _ = send_free.send(p);
            }
            prev = Some(chunk);
        }
        Ok(())
    });

    let upper = opts.uppercase;
    let mut cursor = 0usize;
    while cursor < blocks {
        let n = (blocks - cursor).min(chunk_blocks);
        let mut chunk_out = recv_free.recv().unwrap();
        chunk_out.resize(n * 32 + 16, 0);
        {
            chunk_out[..n * 32]
                .par_chunks_mut(32)
                .enumerate()
                .for_each(|(i, row)| {
                    unsafe {
                        plain_blocks(row.as_mut_ptr(), data.as_ptr().add((cursor + i) * 16), 1, upper);
                    }
                });
        }
        chunk_out.truncate(n * 32);
        send_data.send(chunk_out).unwrap();
        cursor += n;
    }
    drop(send_data);
    writer.join().unwrap()?;

    let mut so = io::stdout().lock();
    if tail_len > 0 {
        let hex = if opts.uppercase { HEX_UPPER } else { HEX_LOWER };
        for &b in &data[blocks * 16..] {
            so.write_all(&[hex[(b >> 4) as usize], hex[(b & 0xf) as usize]])?;
        }
    }
    so.write_all(b"\n")?;
    so.flush()
}

fn run_plain(opts: &Options, reader: &mut dyn Read) -> io::Result<()> {
    let _hex = if opts.uppercase { HEX_UPPER } else { HEX_LOWER };
    let stdout = io::stdout();
    let mut out = BufWriter::with_capacity(WRITE_BUF, stdout.lock());
    let mut buf = vec![0u8; READ_BUF];
    let mut total = 0u64;
    loop {
        let max = match opts.length {
            Some(lim) => buf.len().min((lim.saturating_sub(total)) as usize),
            None => buf.len(),
        };
        if max == 0 { break; }
        let n = reader.read(&mut buf[..max])?;
        if n == 0 { break; }
        total += n as u64;
        for &b in &buf[..n] { out.write_all(&[_hex[(b>>4) as usize], _hex[(b&0xf) as usize]])?; }
    }
    out.write_all(b"\n")?;
    out.flush()
}

fn include_line(dst: *mut u8, src: &[u8], upper: bool) {
    let hex = if upper { HEX_UPPER } else { HEX_LOWER };
    let mut p = dst;
    unsafe {
        *p = b' ';
        *p.add(1) = b' ';
        p = p.add(2);
        for &b in src {
            *p = b'0';
            *p.add(1) = b'x';
            *p.add(2) = hex[(b >> 4) as usize];
            *p.add(3) = hex[(b & 0xf) as usize];
            *p.add(4) = b',';
            *p.add(5) = b' ';
            p = p.add(6);
        }
    }
}

fn run_c_include_mmap(opts: &Options, data: &[u8]) -> io::Result<()> {
    let name = opts.include_name.as_deref().unwrap_or("data");
    let lines = data.len() / 12;
    let tail_len = data.len() % 12;
    let line_len = 75usize;
    let chunk_lines = ((16 * 1024 * 1024) / line_len).max(1);
    let buf_cap = chunk_lines * line_len + 16;

    {
        let mut so = io::stdout().lock();
        writeln!(so, "unsigned char {}[] = {{", name)?;
        so.flush()?;
    }

    let (send_data, recv_data) = sync_channel::<Vec<u8>>(5);
    let (send_free, recv_free) = channel::<Vec<u8>>();
    for _ in 0..6 {
        let buf = vec![0u8; buf_cap];
        send_free.send(buf).unwrap();
    }
    let writer = thread::spawn(move || -> io::Result<()> {
        let mut zc = ZeroCopyWriter::new()?;
        let mut prev: Option<Vec<u8>> = None;
        while let Ok(chunk) = recv_data.recv() {
            zc.write_chunk(&chunk)?;
            if let Some(p) = prev.take() {
                let _ = send_free.send(p);
            }
            prev = Some(chunk);
        }
        Ok(())
    });

    let upper = opts.uppercase;
    let mut cursor = 0usize;
    while cursor < lines {
        let n = (lines - cursor).min(chunk_lines);
        let mut chunk_out = recv_free.recv().unwrap();
        chunk_out.resize(n * line_len + 16, 0);
        {
            chunk_out[..n * line_len]
                .par_chunks_mut(line_len)
                .enumerate()
                .for_each(|(i, row)| {
                    let src = &data[(cursor + i) * 12..(cursor + i) * 12 + 12];
                    include_line(row.as_mut_ptr(), src, upper);
                    unsafe { *row.as_mut_ptr().add(74) = b'\n'; }
                });
        }
        chunk_out.truncate(n * line_len);
        send_data.send(chunk_out).unwrap();
        cursor += n;
    }
    drop(send_data);
    writer.join().unwrap()?;

    {
        let mut so = io::stdout().lock();
        if tail_len > 0 {
            let mut tmp = [0u8; 75];
            include_line(tmp.as_mut_ptr(), &data[lines * 12..], opts.uppercase);
            let tlen = tail_len * 6;
            so.write_all(&tmp[..tlen])?;
        }
        writeln!(so, "\n}};")?;
        writeln!(so, "unsigned int {}_len = {};", name, data.len())?;
        so.flush()
    }
}

fn run_c_include(opts: &Options, reader: &mut dyn Read) -> io::Result<()> {
    let name = opts.include_name.as_deref().unwrap_or("data");
    let _hex = if opts.uppercase { HEX_UPPER } else { HEX_LOWER };
    let stdout = io::stdout();
    let mut out = BufWriter::with_capacity(WRITE_BUF, stdout.lock());
    writeln!(out, "unsigned char {}[] = {{", name)?;
    let mut buf = vec![0u8; READ_BUF];
    let mut first = true;
    let mut col = 0usize;
    let mut total = 0u64;
    loop {
        let max = match opts.length {
            Some(lim) => buf.len().min((lim.saturating_sub(total)) as usize),
            None => buf.len(),
        };
        if max == 0 { break; }
        let n = reader.read(&mut buf[..max])?;
        if n == 0 { break; }
        total += n as u64;
        for &b in &buf[..n] {
            if !first { out.write_all(b", ")?; }
            first = false;
            if col == 12 { out.write_all(b"\n  ")?; col = 0; }
            else if col == 0 { out.write_all(b"  ")?; }
            out.write_all(&[b'0', b'x', _hex[(b>>4) as usize], _hex[(b&0xf) as usize]])?;
            col += 1;
        }
    }
    writeln!(out, "\n}};")?;
    writeln!(out, "unsigned int {}_len = {};", name, total)?;
    out.flush()
}

fn run_reverse(opts: &Options, reader: &mut dyn Read) -> io::Result<()> {
    let stdout = io::stdout();
    let mut out = BufWriter::with_capacity(WRITE_BUF, stdout.lock());
    let jump_off = opts.reverse_jump.unwrap_or(0).max(0) as u64;

    let mut content = Vec::new();
    reader.read_to_end(&mut content)?;

    // Check if the input contains any ':' (canonical mode)
    let is_canonical = content.iter().any(|&b| b == b':');

    let mut output_offset = 0u64;
    let mut last_line_bytes: Vec<u8> = Vec::new();
    let mut squeezing = false;

    for line in content.split(|&b| b == b'\n') {
        if line.is_empty() { continue; }

        if line == b"*" {
            squeezing = true;
            continue;
        }

        let mut pos = 0usize;
        while pos < line.len() && line[pos] == b' ' { pos += 1; }

        let offset_start = pos;
        while pos < line.len() && (line[pos].is_ascii_hexdigit() || line[pos] == b'x') { pos += 1; }
        let offset_end = pos;
        
        let has_sep = pos < line.len() && (line[pos] == b':' || line[pos] == b' ');
        
        // In canonical mode, a line without a separator is the final offset line.
        if is_canonical && !has_sep {
            // Update output_offset in case we were squeezing
            if squeezing && !last_line_bytes.is_empty() {
                let line_off = std::str::from_utf8(&line[offset_start..offset_end]).unwrap_or("");
                let line_off = u64::from_str_radix(line_off, 16).unwrap_or(output_offset);
                let repeat = if last_line_bytes.is_empty() { 0 } else { (line_off - output_offset) / last_line_bytes.len() as u64 };
                for _ in 0..repeat {
                    out.write_all(&last_line_bytes)?;
                }
                output_offset = line_off;
                squeezing = false;
            }
            continue;
        }

        let line_off = if has_sep && offset_end > offset_start {
            let s = std::str::from_utf8(&line[offset_start..offset_end]).unwrap_or("");
            let base = if s.starts_with("0x") { &s[2..] } else { s };
            u64::from_str_radix(base, 16).unwrap_or(output_offset)
        } else { output_offset };

        if line_off < jump_off { continue; }

        if squeezing && !last_line_bytes.is_empty() {
            let repeat = if last_line_bytes.is_empty() { 0 } else { (line_off - output_offset) / last_line_bytes.len() as u64 };
            for _ in 0..repeat {
                out.write_all(&last_line_bytes)?;
            }
            output_offset = line_off;
            squeezing = false;
        }

        if has_sep {
            pos += 1;
        } else {
            pos = offset_start;
        }

        let mut current_line_bytes = Vec::new();

        while pos < line.len() && line[pos] == b' ' { pos += 1; }
        while pos < line.len() && line[pos] != b'|' {
            if line[pos] == b' ' { pos += 1; continue; }
            if pos + 1 < line.len() && line[pos].is_ascii_hexdigit() && line[pos+1].is_ascii_hexdigit() {
                let hi = hex_digit(line[pos]);
                let lo = hex_digit(line[pos+1]);
                let byte = (hi << 4) | lo;
                current_line_bytes.push(byte);
                pos += 2;
            } else { break; }
        }

        if !current_line_bytes.is_empty() {
            out.write_all(&current_line_bytes)?;
            output_offset += current_line_bytes.len() as u64;
            last_line_bytes = current_line_bytes;
        }
    }
    out.flush()
}

fn hex_digit(b: u8) -> u8 {
    match b {
        b'0'..=b'9' => b - b'0',
        b'a'..=b'f' => b - b'a' + 10,
        b'A'..=b'F' => b - b'A' + 10,
        _ => 0,
    }
}

/// Minimal hexdump -e format interpreter.
/// Supports: 'N/M "fmt"' where fmt can have %02x %03o %05d %_c %08_ax %08_Ad
fn run_custom_format(opts: &Options, data: &[u8], start_off: u64) -> io::Result<()> {
    let stdout = io::stdout();
    let mut out = BufWriter::with_capacity(WRITE_BUF, stdout.lock());

    // Parse each format string into a list of units: (count, size, fmt_str)
    struct Unit { count: usize, size: usize, fmt: String }
    let mut units: Vec<Unit> = Vec::new();

    for fmtstr in &opts.formats {
        let s = fmtstr.trim();
        // Pattern: [count/size] "fmt_string" ["sep_string"]
        let mut pos = 0usize;
        while pos < s.len() {
            // skip whitespace
            while pos < s.len() && s.as_bytes()[pos] == b' ' { pos += 1; }
            if pos >= s.len() { break; }

            // check for count/size prefix
            let mut count = 1;
            let mut size = 1;
            let has_prefix = s.as_bytes()[pos].is_ascii_digit();
            if has_prefix {
                let num_end = s[pos..].find(|c: char| !c.is_ascii_digit()).map(|p| pos+p).unwrap_or(s.len());
                count = s[pos..num_end].parse().unwrap_or(1);
                pos = num_end;
                if pos < s.len() && s.as_bytes()[pos] == b'/' {
                    pos += 1;
                    let ne = s[pos..].find(|c: char| !c.is_ascii_digit()).map(|p| pos+p).unwrap_or(s.len());
                    size = s[pos..ne].parse().unwrap_or(1);
                    pos = ne;
                }
                // skip whitespace
                while pos < s.len() && s.as_bytes()[pos] == b' ' { pos += 1; }
            }

            // quoted format string
            if pos < s.len() && s.as_bytes()[pos] == b'"' {
                pos += 1;
                let mut fmt = String::new();
                while pos < s.len() && s.as_bytes()[pos] != b'"' {
                    if s.as_bytes()[pos] == b'\\' {
                        pos += 1;
                        if pos < s.len() {
                            fmt.push(match s.as_bytes()[pos] {
                                b'n' => '\n', b't' => '\t', b'0' => '\0', _ => s.as_bytes()[pos] as char,
                            });
                            pos += 1;
                        }
                    } else {
                        fmt.push(s.as_bytes()[pos] as char);
                        pos += 1;
                    }
                }
                if pos < s.len() { pos += 1; } // closing "
                
                // If no explicit size was given and the string has no format specifiers, size = 0.
                let has_fmt_spec = fmt.contains('%') || fmt.contains("_ax") || fmt.contains("_Ax") || fmt.contains("_Ad");
                if !has_fmt_spec && !has_prefix {
                    size = 0;
                }
                units.push(Unit { count, size, fmt });
            } else {
                pos += 1; // skip unknown char
            }
        }
    }

    let _hex = if opts.uppercase { HEX_UPPER } else { HEX_LOWER };
    let mut data_pos = 0usize;
    let row_size: usize = units.iter().map(|u| u.count * u.size).sum::<usize>().max(1);

    while data_pos < data.len() {
        let mut cur = data_pos;
        for unit in &units {
            for _ in 0..unit.count {
                // apply fmt to cur..cur+unit.size
                let slice = &data[cur.min(data.len())..data.len()];
                let val: u64 = if !slice.is_empty() {
                    let take = unit.size.min(slice.len());
                    read_le_u64(slice, take, Endian::Big)
                } else { 0 };
                let b = if !slice.is_empty() { slice[0] } else { 0 };

                // interpret format specifier
                let fmt = &unit.fmt;
                if fmt.contains("_ax") || fmt.contains("_Ax") {
                    // offset in hex
                    let off = start_off + cur as u64;
                    let s = format!("{:08x}", off);
                    let rest = fmt.split(|c| c == 'x' || c == 'X').last().unwrap_or("");
                    out.write_all(s.as_bytes())?;
                    out.write_all(rest.as_bytes())?;
                } else if fmt.contains("_Ad") {
                    let off = start_off + cur as u64;
                    let s = format!("{:08}", off);
                    let rest = fmt.split('d').last().unwrap_or("");
                    out.write_all(s.as_bytes())?;
                    out.write_all(rest.as_bytes())?;
                } else if fmt.contains("%_c") {
                    let c = if b >= 0x20 && b <= 0x7e { b as char } else { '.' };
                    out.write_all(c.to_string().as_bytes())?;
                } else if fmt.contains("%02x") {
                    let s = format!("{:02x} ", b);
                    out.write_all(s.as_bytes())?;
                } else if fmt.contains("%03o") {
                    let s = format!("{:03o} ", b);
                    out.write_all(s.as_bytes())?;
                } else if fmt.contains("%05d") {
                    let s = format!("{:05} ", val as u16);
                    out.write_all(s.as_bytes())?;
                } else {
                    // literal
                    out.write_all(fmt.as_bytes())?;
                }
                cur += unit.size;
            }
        }
        data_pos += row_size;
    }
    out.flush()
}

fn border_top(out: &mut impl Write, pos_w: usize, hex_w: usize, ascii_w: usize,
              has_pos: bool, has_ascii: bool, border: BorderStyle) -> io::Result<()> {
    match border {
        BorderStyle::None => Ok(()),
        BorderStyle::Ascii => {
            write!(out, "+")?;
            if has_pos { write!(out, "{:-<w$}+", "", w=pos_w)?; }
            if has_ascii {
                write!(out, "{:-<w$}+", "", w=hex_w+1)?;
                write!(out, "{:-<w$}+", "", w=ascii_w+1)?;
            } else {
                write!(out, "{:-<w$}+", "", w=hex_w+1)?;
            }
            writeln!(out)
        }
        BorderStyle::Unicode => {
            write!(out, "┌")?;
            if has_pos { write!(out, "{:─<w$}┬", "", w=pos_w)?; }
            if has_ascii {
                write!(out, "{:─<w$}┬", "", w=hex_w+1)?;
                write!(out, "{:─<w$}┐", "", w=ascii_w+1)?;
            } else {
                write!(out, "{:─<w$}┐", "", w=hex_w+1)?;
            }
            writeln!(out)
        }
    }
}

fn border_header(out: &mut impl Write, pos_w: usize, hex_w: usize, ascii_w: usize,
                 has_pos: bool, has_ascii: bool, border: BorderStyle) -> io::Result<()> {
    match border {
        BorderStyle::None => Ok(()),
        BorderStyle::Ascii => {
            write!(out, "|")?;
            if has_pos { write!(out, "{:^w$}|", "offset", w=pos_w)?; }
            if has_ascii {
                write!(out, "{:^w$}|", " hex", w=hex_w+1)?;
                write!(out, "{:^w$}|", " ascii", w=ascii_w+1)?;
            } else {
                write!(out, "{:^w$}|", " hex", w=hex_w+1)?;
            }
            writeln!(out)
        }
        BorderStyle::Unicode => {
            write!(out, "│")?;
            if has_pos { write!(out, "{:^w$}│", "offset", w=pos_w)?; }
            if has_ascii {
                write!(out, "{:^w$}│", " hex", w=hex_w+1)?;
                write!(out, "{:^w$}│", " ascii", w=ascii_w+1)?;
            } else {
                write!(out, "{:^w$}│", " hex", w=hex_w+1)?;
            }
            writeln!(out)
        }
    }
}

fn border_sep(out: &mut impl Write, pos_w: usize, hex_w: usize, ascii_w: usize,
              has_pos: bool, has_ascii: bool, border: BorderStyle) -> io::Result<()> {
    match border {
        BorderStyle::None => Ok(()),
        BorderStyle::Ascii => {
            write!(out, "+")?;
            if has_pos { write!(out, "{:-<w$}+", "", w=pos_w)?; }
            if has_ascii {
                write!(out, "{:-<w$}+", "", w=hex_w+1)?;
                write!(out, "{:-<w$}+", "", w=ascii_w+1)?;
            } else {
                write!(out, "{:-<w$}+", "", w=hex_w+1)?;
            }
            writeln!(out)
        }
        BorderStyle::Unicode => {
            write!(out, "├")?;
            if has_pos { write!(out, "{:─<w$}┼", "", w=pos_w)?; }
            if has_ascii {
                write!(out, "{:─<w$}┼", "", w=hex_w+1)?;
                write!(out, "{:─<w$}┤", "", w=ascii_w+1)?;
            } else {
                write!(out, "{:─<w$}┤", "", w=hex_w+1)?;
            }
            writeln!(out)
        }
    }
}

fn border_bottom(out: &mut impl Write, pos_w: usize, hex_w: usize, ascii_w: usize,
                 has_pos: bool, has_ascii: bool, border: BorderStyle) -> io::Result<()> {
    match border {
        BorderStyle::None => Ok(()),
        BorderStyle::Ascii => {
            write!(out, "+")?;
            if has_pos { write!(out, "{:-<w$}+", "", w=pos_w)?; }
            if has_ascii {
                write!(out, "{:-<w$}+", "", w=hex_w+1)?;
                write!(out, "{:-<w$}+", "", w=ascii_w+1)?;
            } else {
                write!(out, "{:-<w$}+", "", w=hex_w+1)?;
            }
            writeln!(out)
        }
        BorderStyle::Unicode => {
            write!(out, "└")?;
            if has_pos { write!(out, "{:─<w$}┴", "", w=pos_w)?; }
            if has_ascii {
                write!(out, "{:─<w$}┴", "", w=hex_w+1)?;
                write!(out, "{:─<w$}┘", "", w=ascii_w+1)?;
            } else {
                write!(out, "{:─<w$}┘", "", w=hex_w+1)?;
            }
            writeln!(out)
        }
    }
}

fn output_line(
    out:         &mut impl Write,
    src:         &[u8],
    disp_off:    u64,
    opts:        &Options,
    do_color:    bool,
    _hex_col_w:   usize,
    _ascii_col_w: usize,
) -> io::Result<()> {
    if opts.mode != DisplayMode::Canonical
        || opts.group != 1
        || opts.endian != Endian::Big
        || opts.no_position
        || opts.no_ascii
    {
        let mut tmp = Vec::with_capacity(256);
        format_row_generic(&mut tmp, src, disp_off, opts);
        out.write_all(&tmp)?;
        return Ok(());
    }

    let has_pos   = !opts.no_position;
    let has_ascii = !opts.no_ascii && !matches!(opts.mode,
        DisplayMode::Binary | DisplayMode::OneByteOctal | DisplayMode::TwoByteOctal |
        DisplayMode::OneByteDecimal | DisplayMode::TwoByteDecimal |
        DisplayMode::OneByteChar | DisplayMode::OneByteHex | DisplayMode::TwoByteHex);
    let border    = opts.border;
    let n         = src.len();
    let hex       = if opts.uppercase { HEX_UPPER } else { HEX_LOWER };

    // Row building
    let left_bar  = match border { BorderStyle::None=>"", BorderStyle::Ascii=>"|", BorderStyle::Unicode=>"│" };
    let mid_bar   = left_bar;
    let right_bar = left_bar;

    // Position column
    if has_pos {
        if border != BorderStyle::None { out.write_all(left_bar.as_bytes())?; }
        if do_color { out.write_all(ANSI_CYAN.as_bytes())?; }
        let mut tmp = [0u8; 20];
        let olen = write_offset(&mut tmp, disp_off, opts.offset_dec, opts.uppercase);
        out.write_all(&tmp[..olen])?;
        out.write_all(b":")?;
        if do_color { out.write_all(ANSI_RESET.as_bytes())?; }
        if border != BorderStyle::None {
            let pos_w = if opts.offset_dec { 20 } else { offset_len(u64::MAX) } + 1;
            let pad = pos_w.saturating_sub(olen + 1);
            for _ in 0..pad { out.write_all(b" ")?; }
        } else {
            out.write_all(b" ")?;
        }
    }

    // Hex section
    if border != BorderStyle::None {
        if !has_pos { out.write_all(left_bar.as_bytes())?; }
        else        { out.write_all(mid_bar.as_bytes())?; }
    }

    match opts.mode {
        DisplayMode::Canonical => {
            let half = opts.width / 2;
            for i in 0..opts.width {
                if i == half { out.write_all(b" ")?; }
                if i < n {
                    let b = src[i];
                    if do_color { out.write_all(byte_ansi(b, opts.scheme).as_bytes())?; }
                    out.write_all(&[hex[(b>>4) as usize], hex[(b&0xf) as usize]])?;
                    if do_color { out.write_all(ANSI_RESET.as_bytes())?; }
                    if i < opts.width - 1 { out.write_all(b" ")?; }
                } else {
                    out.write_all(b"  ")?;
                    if i < opts.width - 1 { out.write_all(b" ")?; }
                }
            }
        }
        _ => unreachable!("non-canonical handled above"),
    }

    // ASCII panel
    if has_ascii {
        if border != BorderStyle::None {
            out.write_all(b" ")?;
            out.write_all(mid_bar.as_bytes())?;
        } else {
            out.write_all(b" |")?;
        }
        for i in 0..n {
            let b = src[i];
            if do_color {
                if b >= 0x20 && b <= 0x7e { out.write_all(b"\x1b[32m")?; }
                else                      { out.write_all(ANSI_DIM.as_bytes())?; }
                let mut ab = Vec::new();
                ascii_byte(b, opts.table, &mut ab);
                out.write_all(&ab)?;
                out.write_all(ANSI_RESET.as_bytes())?;
            } else {
                let mut ab = Vec::new();
                ascii_byte(b, opts.table, &mut ab);
                out.write_all(&ab)?;
            }
        }
        let ascii_pad = ascii_width(opts).saturating_sub(n);
        for _ in 0..ascii_pad { out.write_all(b" ")?; }
        if border != BorderStyle::None {
            out.write_all(b" ")?;
            out.write_all(right_bar.as_bytes())?;
        } else {
            out.write_all(b"|")?;
        }
    } else {
        if border != BorderStyle::None {
            out.write_all(b" ")?;
            out.write_all(right_bar.as_bytes())?;
        }
    }

    out.write_all(b"\n")
}

fn use_color(opts: &Options) -> bool {
    match opts.color {
        ColorWhen::Always => true,
        ColorWhen::Never  => false,
        ColorWhen::Auto   => io::stdout().is_terminal(),
    }
}

/// True iff we can use the SIMD canonical fast path (no generic overhead).
fn is_simd_eligible(opts: &Options, do_color: bool) -> bool {
    matches!(opts.mode, DisplayMode::Canonical
        | DisplayMode::OneByteHex
        | DisplayMode::TwoByteHex)
        && matches!(opts.table, CharTable::Ascii | CharTable::Braille)
        && opts.formats.is_empty()
        && (!do_color
            || (opts.scheme == ColorScheme::Default
                && opts.table == CharTable::Ascii
                && opts.border == BorderStyle::None
                && !opts.no_ascii))
}

fn old_simd_eligible(opts: &Options) -> bool {
    opts.mode == DisplayMode::Canonical
        && opts.width == 16
        && opts.group == 1
        && opts.endian == Endian::Big
        && !opts.no_ascii
        && !opts.no_position
        && opts.border == BorderStyle::None
        && !opts.uppercase
        && !opts.offset_dec
        && opts.table == CharTable::Ascii
        && opts.formats.is_empty()
}

fn output_line_diverts(opts: &Options) -> bool {
    opts.mode != DisplayMode::Canonical
        || opts.group != 1
        || opts.endian != Endian::Big
        || opts.no_position
        || opts.no_ascii
}

/// Multi-file concatenated reader.
struct MultiReader {
    files: Vec<String>,
    idx:   usize,
    cur:   Option<Box<dyn Read>>,
}

impl MultiReader {
    fn new(files: Vec<String>) -> io::Result<Self> {
        let mut mr = MultiReader { files, idx: 0, cur: None };
        mr.advance()?;
        Ok(mr)
    }

    fn advance(&mut self) -> io::Result<()> {
        if self.idx >= self.files.len() { self.cur = None; return Ok(()); }
        let name = &self.files[self.idx];
        self.idx += 1;
        self.cur = Some(if name == "-" {
            Box::new(io::stdin()) as Box<dyn Read>
        } else {
            Box::new(File::open(name).map_err(|e| io::Error::new(e.kind(), format!("{}: {}", name, e)))?)
        });
        Ok(())
    }
}

impl Read for MultiReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        loop {
            match self.cur {
                None => return Ok(0),
                Some(ref mut r) => {
                    let n = r.read(buf)?;
                    if n > 0 { return Ok(n); }
                    // EOF on this file, try next
                }
            }
            self.advance()?;
        }
    }
}

fn main() -> io::Result<()> {
    #[cfg(unix)]
    unsafe { libc::signal(libc::SIGPIPE, libc::SIG_DFL); }

    let opts = match parse_args() {
        Ok(o)  => o,
        Err(e) => { eprintln!("fasthex: {}", e); std::process::exit(1); }
    };

    if opts.mode == DisplayMode::Reverse {
        let mut reader: Box<dyn Read> = if opts.files.is_empty() || opts.files == ["-"] {
            Box::new(io::stdin())
        } else {
            Box::new(MultiReader::new(opts.files.clone())?)
        };
        return run_reverse(&opts, &mut reader);
    }

    let do_color = use_color(&opts);

    let single_file: Option<File> = if opts.files.len() == 1 && opts.files[0] != "-" {
        match File::open(&opts.files[0]) {
            Ok(f) => Some(f),
            Err(e) => {
                if !opts.quiet {
                    eprintln!("fasthex: {}: {}", opts.files[0], e);
                }
                std::process::exit(1);
            }
        }
    } else { None };

    if let Some(ref file) = single_file {
        if let Ok(mmap) = unsafe { Mmap::map(file) } {
            // Resolve skip (negative = from end)
            let file_len = mmap.len() as i64;
            let skip_abs: usize = if opts.skip < 0 {
                (file_len + opts.skip).max(0) as usize
            } else {
                (opts.skip as usize).min(mmap.len())
            };

            let mut data = &mmap[skip_abs..];
            if let Some(lim) = opts.length {
                data = &data[..(lim as usize).min(data.len())];
            }
            if data.is_empty() { return Ok(()); }

            #[cfg(unix)]
            unsafe {
                libc::madvise(data.as_ptr() as *mut libc::c_void, data.len(), libc::MADV_SEQUENTIAL);
            }

            // Display offset = file position + jump bias
            let start_disp: u64 = (skip_abs as i64 + opts.jump) as u64;

            // Plain / CInclude / custom format
            match opts.mode {
                DisplayMode::Plain => {
                    return run_plain_mmap(&opts, data);
                }
                DisplayMode::CInclude => {
                    return run_c_include_mmap(&opts, data);
                }
                _ => {}
            }
            if !opts.formats.is_empty() {
                return run_custom_format(&opts, data, start_disp);
            }

            let simd_ok = is_simd_eligible(&opts, do_color);
            let avx2_hw = cpu_avx2();
            let use_avx2 = simd_ok && avx2_hw;
            let use_simd = simd_ok && (use_avx2 || cpu_sse41());
            let use_avx512 = use_simd && cpu_avx512();

            let colored_simd = do_color
                && opts.scheme == ColorScheme::Default
                && opts.table == CharTable::Ascii
                && opts.border == BorderStyle::None
                && !opts.no_ascii;
            let serial_only = opts.squeeze || opts.max_lines.is_some();
            if serial_only {
                return run_serial_mmap(&opts, data, start_disp, do_color, use_simd, use_avx2);
            }
            let octal_fast = opts.mode == DisplayMode::OneByteOctal
                && opts.width == 16
                && opts.border == BorderStyle::None
                && !opts.no_position
                && !opts.offset_dec
                && avx2_hw
                && start_disp.wrapping_add(data.len() as u64) <= 0xFFFF_FFFF;
            if octal_fast || (use_simd && (!do_color || colored_simd)) {
                return run_parallel_mmap(&opts, data, start_disp, use_avx2, use_avx512);
            }
            return run_parallel_scalar(&opts, data, start_disp, do_color);
        }
    }

    // Build reader
    let files = if opts.files.is_empty() { vec!["-".to_string()] } else { opts.files.clone() };
    let mut reader: Box<dyn Read> = match MultiReader::new(files) {
        Ok(r) => Box::new(r),
        Err(e) => {
            if !opts.quiet { eprintln!("fasthex: {}", e); }
            std::process::exit(1);
        }
    };

    // skip
    if opts.skip != 0 {
        // Try seek on the first file if single
        if let Some(mut f) = single_file {
            if opts.skip >= 0 {
                f.seek(SeekFrom::Start(opts.skip as u64))?;
                reader = Box::new(f);
            }
        } else if opts.skip > 0 {
            let mut skip_buf = vec![0u8; 8192];
            let mut to_skip = opts.skip as u64;
            while to_skip > 0 {
                let chunk = to_skip.min(skip_buf.len() as u64) as usize;
                let n = reader.read(&mut skip_buf[..chunk])?;
                if n == 0 { break; }
                to_skip -= n as u64;
            }
        }
    }

    match opts.mode {
        DisplayMode::Plain    => return run_plain(&opts, &mut reader),
        DisplayMode::CInclude => return run_c_include(&opts, &mut reader),
        _ => {}
    }

    let simd_ok   = is_simd_eligible(&opts, do_color);
    let use_avx2  = simd_ok && cpu_avx2();
    let use_simd  = simd_ok && (use_avx2 || cpu_sse41());

    run_streaming(&opts, &mut reader, do_color, use_simd, use_avx2)
}

fn row_len_delta(opts: &Options, src: &[u8]) -> i64 {
    match opts.mode {
        DisplayMode::OneByteChar => src.iter().map(|&b| match b {
            0x00 | 0x07..=0x0d => -1i64,
            _ => 0,
        }).sum(),
        _ => match opts.table {
            CharTable::Cp437 => src.iter().map(|&b| CP437[b as usize].len_utf8() as i64 - 1).sum(),
            CharTable::Default => src.iter().map(|&b| {
                if b == 0x00 { 2 } else if b == 0x20 || (0x21..=0x7e).contains(&b) { 0 } else { 2 }
            }).sum(),
            _ => 0,
        },
    }
}

fn run_parallel_scalar(
    opts:      &Options,
    data:      &[u8],
    start_off: u64,
    do_color:  bool,
) -> io::Result<()> {
    let width     = opts.width;
    let file_sz   = data.len();
    let full_rows = file_sz / width;
    let tail_len  = file_sz % width;
    let scalar_kind = if opts.minimal { LayoutKind::Generic } else { LayoutKind::OutputLine };
    let cfg       = RowCfg::new(opts, scalar_kind, scalar_kind);
    let core      = &cfg.core;

    let pos_w = if opts.no_position { 0 } else {
        (if opts.offset_dec { 20 } else { offset_len(u64::MAX) }) + 1
    };
    let hex_w   = hex_width(opts);
    let ascii_w = ascii_width(opts);
    let has_ascii = !opts.no_ascii && !matches!(opts.mode, DisplayMode::Binary | DisplayMode::OneByteOctal | DisplayMode::TwoByteOctal | DisplayMode::OneByteDecimal | DisplayMode::TwoByteDecimal | DisplayMode::OneByteChar | DisplayMode::OneByteHex | DisplayMode::TwoByteHex);
    if opts.border != BorderStyle::None {
        let mut so = io::stdout().lock();
        border_top(&mut so, pos_w, hex_w, ascii_w, !opts.no_position, has_ascii, opts.border)?;
        border_header(&mut so, pos_w, hex_w, ascii_w, !opts.no_position, has_ascii, opts.border)?;
        border_sep(&mut so, pos_w, hex_w, ascii_w, !opts.no_position, has_ascii, opts.border)?;
        so.flush()?;
    }

    let field_len_at = |row: usize| field_len_for(opts, start_off.wrapping_add((row * width) as u64));
    let row_len_at   = |row: usize| core.prefix_len(field_len_at(row)) + core.full.emitted;

    let mut bounds = vec![0usize, full_rows];
    {
        let mut push = |v: u64| {
            if v > start_off {
                let r = ((v - start_off) as usize).div_ceil(width);
                if r > 0 && r < full_rows { bounds.push(r); }
            }
        };
        if opts.offset_dec {
            for k in 8..=19u32 { push(10u64.pow(k)); }
        } else {
            for k in 8..=15u32 { push(1u64 << (4 * k)); }
        }
    }
    bounds.sort_unstable();
    bounds.dedup();

    let first = row_len_at(0);
    let last  = if full_rows > 0 { row_len_at(full_rows - 1) } else { first };
    let nbuf = 6usize;
    let chunk_rows = ((16 * 1024 * 1024) / first.min(last)).max(1);
    let buf_cap    = chunk_rows * first.max(last) + 16;

    #[cfg(unix)]
    unsafe {
        libc::madvise(data.as_ptr() as *mut libc::c_void, data.len(), libc::MADV_WILLNEED);
    }

    let (send_data, recv_data) = sync_channel::<Vec<u8>>(nbuf - 1);
    let (send_free, recv_free) = channel::<Vec<u8>>();
    for _ in 0..nbuf {
        let buf = vec![0u8; buf_cap];
        #[cfg(unix)]
        unsafe {
            #[cfg(target_os = "linux")]
            libc::madvise(buf.as_ptr() as *mut libc::c_void, buf.len(), libc::MADV_HUGEPAGE);
        }
        send_free.send(buf).unwrap();
    }

    let writer = thread::spawn(move || -> io::Result<()> {
        let mut zc = ZeroCopyWriter::new()?;
        let mut prev: Option<Vec<u8>> = None;
        while let Ok(chunk) = recv_data.recv() {
            zc.write_chunk(&chunk)?;
            if let Some(p) = prev.take() {
                let _ = send_free.send(p);
            }
            prev = Some(chunk);
        }
        Ok(())
    });

    let colored_row = !output_line_diverts(opts);
    let variable = opts.mode == DisplayMode::OneByteChar
        || ((opts.mode == DisplayMode::Canonical && !opts.no_ascii)
            && matches!(opts.table, CharTable::Cp437 | CharTable::Default));

    for s in 0..bounds.len() - 1 {
        let seg_start = bounds[s];
        let seg_end   = bounds[s + 1];
        let row_len   = row_len_at(seg_start);
        let mut row_cursor = seg_start;
        while row_cursor < seg_end {
            let rows = (seg_end - row_cursor).min(chunk_rows);
            let mut chunk_out = recv_free.recv().unwrap();
            if variable {
                let deltas: Vec<i64> = (row_cursor..row_cursor + rows)
                    .into_par_iter()
                    .map(|r| row_len_delta(opts, &data[r * width..r * width + width]))
                    .collect();
                let mut offsets = vec![0i64; rows + 1];
                for i in 0..rows {
                    offsets[i + 1] = offsets[i] + row_len as i64 + deltas[i];
                }
                let payload = offsets[rows] as usize;
                chunk_out.resize(payload + 16, 0);
                {
                    let out_ptr = chunk_out.as_mut_ptr() as usize;
                    (0..rows).into_par_iter()
                        .fold(|| Vec::with_capacity(128), |mut v, i| {
                            v.clear();
                            let src_off = (row_cursor + i) * width;
                            let off = start_off.wrapping_add(src_off as u64);
                            if colored_row {
                                output_line(&mut v, &data[src_off..src_off + width], off, opts, do_color, 0, 0).unwrap();
                            } else {
                                format_row_generic(&mut v, &data[src_off..src_off + width], off, opts);
                            }
                            let o0 = offsets[i] as usize;
                            unsafe {
                                std::slice::from_raw_parts_mut((out_ptr + o0) as *mut u8, v.len()).copy_from_slice(&v);
                            }
                            v
                        })
                        .for_each(|_| {});
                }
                chunk_out.truncate(payload);
            } else {
                let payload = rows * row_len;
                chunk_out.resize(payload + 16, 0);
                {
                    let out_rows = &mut chunk_out[..payload];
                    out_rows
                        .par_chunks_mut(row_len)
                        .enumerate()
                        .fold(|| Vec::with_capacity(row_len + 16), |mut v, (i, row)| {
                            v.clear();
                            let src_off = (row_cursor + i) * width;
                            let off = start_off.wrapping_add(src_off as u64);
                            if colored_row {
                                output_line(&mut v, &data[src_off..src_off + width], off, opts, do_color, 0, 0).unwrap();
                            } else {
                                format_row_generic(&mut v, &data[src_off..src_off + width], off, opts);
                            }
                            row[..v.len().min(row_len)].copy_from_slice(&v[..v.len().min(row_len)]);
                            v
                        })
                        .for_each(|_| {});
                }
                chunk_out.truncate(payload);
            }
            send_data.send(chunk_out).unwrap();
            row_cursor += rows;
        }
    }

    drop(send_data);
    writer.join().unwrap()?;

    let mut so = io::stdout().lock();
    if tail_len > 0 {
        let src_off = full_rows * width;
        let off     = start_off.wrapping_add(src_off as u64);
        let mut v = Vec::with_capacity(256);
        if colored_row {
            output_line(&mut v, &data[src_off..], off, opts, do_color, 0, 0)?;
        } else {
            format_row_generic(&mut v, &data[src_off..], off, opts);
        }
        so.write_all(&v)?;
    }

    if opts.border != BorderStyle::None {
        border_bottom(&mut so, pos_w, hex_w, ascii_w, !opts.no_position, has_ascii, opts.border)?;
    } else if !opts.no_position {
        let final_off = start_off.wrapping_add(file_sz as u64);
        let mut tmp = [0u8; 20];
        let olen = write_offset(&mut tmp, final_off, opts.offset_dec, opts.uppercase);
        if do_color { so.write_all(ANSI_CYAN.as_bytes())?; }
        so.write_all(&tmp[..olen])?;
        if do_color { so.write_all(ANSI_RESET.as_bytes())?; }
        so.write_all(b"\n")?;
    }

    so.flush()
}

fn run_parallel_mmap(
    opts:       &Options,
    data:       &[u8],
    start_off:  u64,
    use_avx2:   bool,
    use_avx512: bool,
) -> io::Result<()> {
    let width     = opts.width;
    let file_sz   = data.len();
    let full_rows = file_sz / width;
    let tail_len  = file_sz % width;
    let full_kind = if opts.minimal {
        LayoutKind::Generic
    } else if use_color(opts) || !old_simd_eligible(opts) {
        LayoutKind::OutputLine
    } else {
        LayoutKind::Generic
    };
    let tail_kind = if opts.minimal {
        LayoutKind::Generic
    } else if use_color(opts) || !old_simd_eligible(opts) {
        LayoutKind::OutputLine
    } else {
        LayoutKind::Scalar
    };
    let cfg       = RowCfg::new(opts, full_kind, tail_kind);
    let core      = &cfg.core;
    let blocks    = core.blocks;

    let pos_w = if opts.no_position { 0 } else {
        (if opts.offset_dec { 20 } else { offset_len(u64::MAX) }) + 1
    };
    let hex_w   = hex_width(opts);
    let ascii_w = ascii_width(opts);
    let has_ascii = !opts.no_ascii && !matches!(opts.mode, DisplayMode::Binary | DisplayMode::OneByteOctal | DisplayMode::TwoByteOctal | DisplayMode::OneByteDecimal | DisplayMode::TwoByteDecimal | DisplayMode::OneByteChar | DisplayMode::OneByteHex | DisplayMode::TwoByteHex);
    if opts.border != BorderStyle::None {
        let mut so = io::stdout().lock();
        border_top(&mut so, pos_w, hex_w, ascii_w, !opts.no_position, has_ascii, opts.border)?;
        border_header(&mut so, pos_w, hex_w, ascii_w, !opts.no_position, has_ascii, opts.border)?;
        border_sep(&mut so, pos_w, hex_w, ascii_w, !opts.no_position, has_ascii, opts.border)?;
        so.flush()?;
    }

    let field_len_at = |row: usize| field_len_for(opts, start_off.wrapping_add((row * width) as u64));
    let row_len_at   = |row: usize| core.prefix_len(field_len_at(row)) + core.full.emitted;

    let mut bounds = vec![0usize, full_rows];
    {
        let mut push = |v: u64| {
            if v > start_off {
                let r = ((v - start_off) as usize).div_ceil(width);
                if r > 0 && r < full_rows { bounds.push(r); }
            }
        };
        if opts.offset_dec {
            for k in 8..=19u32 { push(10u64.pow(k)); }
        } else {
            for k in 8..=15u32 { push(1u64 << (4 * k)); }
        }
    }
    bounds.sort_unstable();
    bounds.dedup();

    let first = row_len_at(0);
    let last  = if full_rows > 0 { row_len_at(full_rows - 1) } else { first };
    let nbuf = 6usize;
    let chunk_rows = ((16 * 1024 * 1024) / first.min(last)).max(1);
    let buf_cap    = chunk_rows * first.max(last) + 16;

    #[cfg(unix)]
    unsafe {
        libc::madvise(data.as_ptr() as *mut libc::c_void, data.len(), libc::MADV_WILLNEED);
    }

    let (send_data, recv_data) = sync_channel::<Vec<u8>>(nbuf - 1);
    let (send_free, recv_free) = channel::<Vec<u8>>();
    for _ in 0..nbuf {
        let buf = vec![0u8; buf_cap];
        #[cfg(unix)]
        unsafe {
            #[cfg(target_os = "linux")]
            libc::madvise(buf.as_ptr() as *mut libc::c_void, buf.len(), libc::MADV_HUGEPAGE);
        }
        send_free.send(buf).unwrap();
    }

    let writer = thread::spawn(move || -> io::Result<()> {
        let mut zc = ZeroCopyWriter::new()?;
        let mut prev: Option<Vec<u8>> = None;
        while let Ok(chunk) = recv_data.recv() {
            zc.write_chunk(&chunk)?;
            if let Some(p) = prev.take() {
                let _ = send_free.send(p);
            }
            prev = Some(chunk);
        }
        Ok(())
    });

    let use_pairs = use_avx2 && width == 16 && opts.table == CharTable::Ascii
        && core.full.colored.is_none()
        && (core.full.fast.is_some()
            || (core.full.windows.len() == 4 && core.full.ascii_run >= 16));
    let octal_all = opts.mode == DisplayMode::OneByteOctal
        && opts.width == 16
        && opts.border == BorderStyle::None
        && !opts.no_position
        && !opts.offset_dec
        && start_off.wrapping_add(file_sz as u64) <= 0xFFFF_FFFF;
    let use_octal = octal_all && cpu_avx2();
    let min_all = opts.minimal
        && width == 16
        && opts.table == CharTable::Ascii
        && !opts.no_position
        && !opts.offset_dec
        && !use_color(opts)
        && start_off.wrapping_add(file_sz as u64) <= 0xFFFF_FFFF;
    let use_min = min_all && use_avx2;
    let use_fours = use_avx512 && width == 16 && opts.table == CharTable::Ascii
        && core.full.fast.is_none() && core.full.colored.is_none()
        && core.full.windows.len() == 4 && core.full.ascii_run >= 16;
    let do_color = use_color(opts);

    for s in 0..bounds.len() - 1 {
        let seg_start = bounds[s];
        let seg_end   = bounds[s + 1];
        let row_len   = row_len_at(seg_start);
        let mut row_cursor = seg_start;
        while row_cursor < seg_end {
            let rows = (seg_end - row_cursor).min(chunk_rows);

            let mut chunk_out = recv_free.recv().unwrap();
            let payload = rows * row_len;
            chunk_out.resize(payload + 16, 0);
            {
                let out_rows = &mut chunk_out[..payload];
                if use_min {
                    let row_len_m = if opts.no_ascii { 42usize } else { 59usize };
                    let lut = if opts.uppercase { HEX_UPPER } else { HEX_LOWER };
                    let even = rows & !1;
                    let run_pairs = even / 256;
                    let run_end = run_pairs * 256;
                    out_rows[..run_end * row_len_m]
                        .par_chunks_mut(row_len_m * 256)
                        .enumerate()
                        .for_each(|(i, block)| {
                            let src_off = (row_cursor + i * 256) * width;
                            let off = start_off.wrapping_add(src_off as u64);
                            unsafe {
                                format_pairs_min_run(block.as_mut_ptr(), data.as_ptr().add(src_off),
                                                     off, 128, lut, !opts.no_ascii, row_len_m);
                            }
                        });
                    if even > run_end {
                        let src_off = (row_cursor + run_end) * width;
                        let off = start_off.wrapping_add(src_off as u64);
                        unsafe {
                            format_pairs_min_run(out_rows[run_end * row_len_m..].as_mut_ptr(), data.as_ptr().add(src_off),
                                                 off, (even - run_end) / 2, lut, !opts.no_ascii, row_len_m);
                        }
                    }
                    if rows & 1 != 0 {
                        let i = rows - 1;
                        let src_off = (row_cursor + i) * width;
                        let off = start_off.wrapping_add(src_off as u64);
                        unsafe {
                            format_row(out_rows[i * row_len_m..].as_mut_ptr(),
                                       data.as_ptr().add(src_off), off, width, core, &core.full);
                        }
                    }
                } else if use_octal {
                    let row_len_o = 75usize;
                    let even = rows;
                    let run_rows = even / 256;
                    let run_end = run_rows * 256;
                    out_rows[..run_end * row_len_o]
                        .par_chunks_mut(row_len_o * 256)
                        .enumerate()
                        .for_each(|(i, block)| {
                            let src_off = (row_cursor + i * 256) * width;
                            let off = start_off.wrapping_add(src_off as u64);
                            unsafe {
                                format_octal_run(block.as_mut_ptr(), data.as_ptr().add(src_off),
                                                 off, 256, opts.uppercase, row_len_o);
                            }
                        });
                    if even > run_end {
                        let src_off = (row_cursor + run_end) * width;
                        let off = start_off.wrapping_add(src_off as u64);
                        unsafe {
                            format_octal_run(out_rows[run_end * row_len_o..].as_mut_ptr(), data.as_ptr().add(src_off),
                                             off, even - run_end, opts.uppercase, row_len_o);
                        }
                    }
                } else if use_fours {
                    let (idx, sp) = pair_masks(&core.full);
                    let last_off = start_off.wrapping_add(((row_cursor + rows - 1) * width) as u64);
                    let fast_all = core.opts.border == BorderStyle::None
                        && !core.opts.no_position
                        && !core.opts.offset_dec
                        && last_off.wrapping_add(48) <= 0xFFFF_FFFF;
                    let lut = if opts.uppercase { HEX_UPPER } else { HEX_LOWER };
                    let quads = rows / 4;
                    let quad_end = quads * 4;
                    let run_quads = quads / 2;
                    let run_end = run_quads * 8;
                    out_rows[..run_end * row_len]
                        .par_chunks_mut(row_len * 8)
                        .enumerate()
                        .for_each(|(i, block)| {
                            let src_off = (row_cursor + i * 8) * width;
                            let off = start_off.wrapping_add(src_off as u64);
                            unsafe {
                                format_fours_run(block.as_mut_ptr(), data.as_ptr().add(src_off),
                                                 off, 2, &idx, &sp, fast_all, lut, core, &core.full, row_len);
                            }
                        });
                    if quad_end > run_end {
                        let src_off = (row_cursor + run_end) * width;
                        let off = start_off.wrapping_add(src_off as u64);
                        unsafe {
                            format_fours_run(out_rows[run_end * row_len..].as_mut_ptr(), data.as_ptr().add(src_off),
                                             off, (quad_end - run_end) / 4, &idx, &sp, fast_all, lut, core, &core.full, row_len);
                        }
                    }
                    if rows - quad_end >= 2 {
                        let src_off = (row_cursor + quad_end) * width;
                        let off = start_off.wrapping_add(src_off as u64);
                        if use_avx2 {
                            let fast_off = fast_offsets_ok(core, off);
                            let o8 = if fast_off {
                                unsafe { hex_offsets_4(off, lut) }
                            } else { [0; 4] };
                            unsafe {
                                format_pair(out_rows[quad_end * row_len..].as_mut_ptr(), data.as_ptr().add(src_off),
                                            off, [o8[0], o8[1]], fast_off, &idx, &sp, core, &core.full, row_len);
                            }
                        } else {
                            unsafe {
                                format_row(out_rows[quad_end * row_len..].as_mut_ptr(),
                                           data.as_ptr().add(src_off), off, width, core, &core.full);
                                format_row(out_rows[(quad_end + 1) * row_len..].as_mut_ptr(),
                                           data.as_ptr().add(src_off + 16), off.wrapping_add(16), width, core, &core.full);
                            }
                        }
                    }
                    if rows & 1 != 0 {
                        let i = rows - 1;
                        let src_off = (row_cursor + i) * width;
                        let off = start_off.wrapping_add(src_off as u64);
                        unsafe {
                            format_row(out_rows[i * row_len..].as_mut_ptr(),
                                       data.as_ptr().add(src_off), off, width, core, &core.full);
                        }
                    }
                } else if use_pairs {
                    let last_off = start_off.wrapping_add(((row_cursor + rows - 1) * width) as u64);
                    let fast_all = core.opts.border == BorderStyle::None
                        && !core.opts.no_position
                        && !core.opts.offset_dec
                        && last_off.wrapping_add(48) <= 0xFFFF_FFFF;
                    let lut = if opts.uppercase { HEX_UPPER } else { HEX_LOWER };
                    let even = rows & !1;
                    let run_pairs = even / 256;
                    let run_end = run_pairs * 256;
                    if fast_all && core.full.fast.is_some() {
                        let k = core.full.fast.as_ref().unwrap();
                        out_rows[..run_end * row_len]
                            .par_chunks_mut(row_len * 256)
                            .enumerate()
                            .for_each(|(i, block)| {
                                let src_off = (row_cursor + i * 256) * width;
                                let off = start_off.wrapping_add(src_off as u64);
                                unsafe {
                                    format_pairs_fast_run(block.as_mut_ptr(), data.as_ptr().add(src_off),
                                                          off, 128, lut, *k, row_len);
                                }
                            });
                        if even > run_end {
                            let src_off = (row_cursor + run_end) * width;
                            let off = start_off.wrapping_add(src_off as u64);
                            unsafe {
                                format_pairs_fast_run(out_rows[run_end * row_len..].as_mut_ptr(), data.as_ptr().add(src_off),
                                                      off, (even - run_end) / 2, lut, *k, row_len);
                            }
                        }
                        if rows & 1 != 0 {
                            let i = rows - 1;
                            let src_off = (row_cursor + i) * width;
                            let off = start_off.wrapping_add(src_off as u64);
                            unsafe {
                                format_row(out_rows[i * row_len..].as_mut_ptr(),
                                           data.as_ptr().add(src_off), off, width, core, &core.full);
                            }
                        }
                    } else {
                        let (idx, sp) = pair_masks(&core.full);
                        out_rows[..run_end * row_len]
                            .par_chunks_mut(row_len * 8)
                            .enumerate()
                            .for_each(|(i, block)| {
                                let src_off = (row_cursor + i * 8) * width;
                                let off = start_off.wrapping_add(src_off as u64);
                                unsafe {
                                    format_pairs_run(block.as_mut_ptr(), data.as_ptr().add(src_off),
                                                     off, 4, &idx, &sp, fast_all, lut, core, &core.full, row_len);
                                }
                            });
                        if even > run_end {
                            let src_off = (row_cursor + run_end) * width;
                            let off = start_off.wrapping_add(src_off as u64);
                            unsafe {
                                format_pairs_run(out_rows[run_end * row_len..].as_mut_ptr(), data.as_ptr().add(src_off),
                                                 off, (even - run_end) / 2, &idx, &sp, fast_all, lut, core, &core.full, row_len);
                            }
                        }
                        if rows & 1 != 0 {
                            let i = rows - 1;
                            let src_off = (row_cursor + i) * width;
                            let off = start_off.wrapping_add(src_off as u64);
                            unsafe {
                                format_row(out_rows[i * row_len..].as_mut_ptr(),
                                           data.as_ptr().add(src_off), off, width, core, &core.full);
                            }
                        }
                    }
                } else {
                    out_rows
                        .par_chunks_mut(row_len)
                        .enumerate()
                        .for_each(|(i, row)| {
                            let src_off = (row_cursor + i) * width;
                            let off = start_off.wrapping_add(src_off as u64);
                            let src_ptr;
                            let mut tmp;
                            if src_off + blocks * 16 <= file_sz {
                                src_ptr = unsafe { data.as_ptr().add(src_off) };
                            } else {
                                tmp = vec![0u8; blocks * 16];
                                tmp[..width].copy_from_slice(&data[src_off..src_off + width]);
                                src_ptr = tmp.as_ptr();
                            }
                            unsafe {
                                format_row(row.as_mut_ptr(), src_ptr, off, width, core, &core.full);
                            }
                        });
                }
            }
            chunk_out.truncate(payload);
            send_data.send(chunk_out).unwrap();
            row_cursor += rows;
        }
    }

    drop(send_data);
    writer.join().unwrap()?;

    let mut so = io::stdout().lock();
    if tail_len > 0 {
        let src_off = full_rows * width;
        let off     = start_off.wrapping_add(src_off as u64);
        if use_octal || use_min {
            let mut v = Vec::with_capacity(128);
            format_row_generic(&mut v, &data[src_off..], off, opts);
            so.write_all(&v)?;
        } else {
            let mut tmp = vec![0u8; blocks * 16];
            tmp[..tail_len].copy_from_slice(&data[src_off..]);
            let layout = cfg.layout(tail_len);
            let cap = core.prefix_len(field_len_at(full_rows)) + layout.emitted + 16;
            let mut v = vec![0u8; cap];
            let n = unsafe { format_row(v.as_mut_ptr(), tmp.as_ptr(), off, tail_len, core, layout) };
            so.write_all(&v[..n])?;
        }
    }

    if opts.border != BorderStyle::None {
        border_bottom(&mut so, pos_w, hex_w, ascii_w, !opts.no_position, has_ascii, opts.border)?;
    } else if !opts.no_position {
        let final_off = start_off.wrapping_add(file_sz as u64);
        let mut tmp = [0u8; 20];
        let olen = write_offset(&mut tmp, final_off, opts.offset_dec, opts.uppercase);
        if do_color { so.write_all(ANSI_CYAN.as_bytes())?; }
        so.write_all(&tmp[..olen])?;
        if do_color { so.write_all(ANSI_RESET.as_bytes())?; }
        so.write_all(b"\n")?;
    }

    so.flush()
}

fn run_serial_mmap(
    opts:      &Options,
    data:      &[u8],
    start_off: u64,
    do_color:  bool,
    use_simd:  bool,
    use_avx2:  bool,
) -> io::Result<()> {
    let bpr       = opts.width;
    let file_sz   = data.len();
    let full_rows = file_sz / bpr;
    let tail_len  = file_sz % bpr;

    let stdout = io::stdout();
    let mut out = BufWriter::with_capacity(WRITE_BUF, stdout.lock());

    // Border top
    let pos_w = if opts.no_position { 0 } else {
        (if opts.offset_dec { 20 } else { offset_len(u64::MAX) }) + 1
    };
    let hex_w   = hex_width(opts);
    let ascii_w = ascii_width(opts);
    let has_ascii = !opts.no_ascii && !matches!(opts.mode, DisplayMode::Binary | DisplayMode::OneByteOctal | DisplayMode::TwoByteOctal | DisplayMode::OneByteDecimal | DisplayMode::TwoByteDecimal | DisplayMode::OneByteChar | DisplayMode::OneByteHex | DisplayMode::TwoByteHex);
    if opts.border != BorderStyle::None {
        border_top(&mut out, pos_w, hex_w, ascii_w, !opts.no_position,
                   has_ascii, opts.border)?;
        border_header(&mut out, pos_w, hex_w, ascii_w, !opts.no_position,
                      has_ascii, opts.border)?;
        border_sep(&mut out, pos_w, hex_w, ascii_w, !opts.no_position,
                   has_ascii, opts.border)?;
    }

    let mut prev_row: Vec<u8>  = Vec::new();
    let mut squeezed            = false;
    let mut lines_written: u64  = 0;
    let simd_ok                 = is_simd_eligible(opts, do_color);
    let full_kind = if opts.minimal {
        LayoutKind::Generic
    } else if do_color || !old_simd_eligible(opts) {
        LayoutKind::OutputLine
    } else {
        LayoutKind::Generic
    };
    let cfg                     = if simd_ok && use_simd {
        Some(RowCfg::new(opts, full_kind, if opts.minimal { LayoutKind::Generic } else { LayoutKind::OutputLine }))
    } else { None };
    let core                    = cfg.as_ref().map(|c| &c.core);
    let max_field               = field_len_for(opts, start_off.wrapping_add(file_sz as u64));
    let row_cap                 = core.map_or(16, |c| (c.prefix_len(max_field) + c.full.emitted) * 16 + 32);
    let mut row_buf             = vec![0u8; row_cap];
    let mut scratch             = vec![0u8; core.map_or(0, |c| c.blocks * 16).max(32)];
    let pair_ok                 = core.map_or(false, can_pair)
        && use_avx2 && !opts.squeeze && opts.max_lines.is_none();
    let min_ser = use_avx2 && opts.minimal && bpr == 16
        && !opts.no_position && !opts.offset_dec
        && start_off.wrapping_add(((full_rows.saturating_sub(1)) * bpr) as u64) <= 0xFFFF_FFFF;
    let run_ok                  = (pair_ok && core.and_then(|c| c.full.fast.as_ref()).is_some() || min_ser)
        && !opts.offset_dec
        && start_off.wrapping_add(((full_rows.saturating_sub(1)) * bpr) as u64) <= 0xFFFF_FFFF;

    if run_ok {
        let core = core.unwrap();
        let lut = if opts.uppercase { HEX_UPPER } else { HEX_LOWER };
        let row_len = if opts.minimal {
            if opts.no_ascii { 42 } else { 59 }
        } else {
            core.prefix_len(8) + core.full.emitted
        };
        let mut r = 0usize;
        while r + 16 <= full_rows {
            if opts.minimal {
                unsafe {
                    format_pairs_min_run(row_buf.as_mut_ptr(), data.as_ptr().add(r * bpr),
                                         start_off.wrapping_add((r * bpr) as u64), 8, lut, !opts.no_ascii, row_len);
                }
            } else {
                let k = core.full.fast.as_ref().unwrap();
                unsafe {
                    format_pairs_fast_run(row_buf.as_mut_ptr(), data.as_ptr().add(r * bpr),
                                          start_off.wrapping_add((r * bpr) as u64), 8, lut, *k, row_len);
                }
            }
            out.write_all(&row_buf[..row_len * 16])?;
            r += 16;
        }
        let rem = full_rows - r;
        if rem >= 2 {
            let pairs = rem / 2;
            if opts.minimal {
                unsafe {
                    format_pairs_min_run(row_buf.as_mut_ptr(), data.as_ptr().add(r * bpr),
                                         start_off.wrapping_add((r * bpr) as u64), pairs, lut, !opts.no_ascii, row_len);
                }
            } else {
                let k = core.full.fast.as_ref().unwrap();
                unsafe {
                    format_pairs_fast_run(row_buf.as_mut_ptr(), data.as_ptr().add(r * bpr),
                                          start_off.wrapping_add((r * bpr) as u64), pairs, lut, *k, row_len);
                }
            }
            out.write_all(&row_buf[..row_len * pairs * 2])?;
            r += pairs * 2;
        }
        if r < full_rows {
            let disp_off = start_off.wrapping_add((r * bpr) as u64);
            let src_ptr = if r * bpr + core.blocks * 16 <= file_sz {
                unsafe { data.as_ptr().add(r * bpr) }
            } else {
                scratch[..bpr].copy_from_slice(&data[r * bpr..]);
                scratch.as_ptr()
            };
            let n = unsafe { format_row(row_buf.as_mut_ptr(), src_ptr, disp_off, bpr, core, &core.full) };
            out.write_all(&row_buf[..n])?;
        }
        if tail_len > 0 && opts.max_lines.map_or(true, |m| lines_written < m) {
            let src_off  = full_rows * bpr;
            let disp_off = start_off.wrapping_add(src_off as u64);
            scratch[..tail_len].copy_from_slice(&data[src_off..]);
            let layout = cfg.as_ref().unwrap().layout(tail_len);
            let n = unsafe { format_row(row_buf.as_mut_ptr(), scratch.as_ptr(), disp_off, tail_len, core, layout) };
            out.write_all(&row_buf[..n])?;
        }
        if opts.border != BorderStyle::None {
            border_bottom(&mut out, pos_w, hex_w, ascii_w, !opts.no_position,
                          has_ascii, opts.border)?;
        } else if !opts.no_position {
            let final_off = start_off.wrapping_add(file_sz as u64);
            let mut tmp = [0u8; 20];
            let olen = write_offset(&mut tmp, final_off, opts.offset_dec, opts.uppercase);
            if do_color { out.write_all(ANSI_CYAN.as_bytes())?; }
            out.write_all(&tmp[..olen])?;
            if do_color { out.write_all(ANSI_RESET.as_bytes())?; }
            out.write_all(b"\n")?;
        }
        return out.flush();
    }

    let mut r = 0usize;
    while r < full_rows {
        if opts.max_lines.map_or(false, |m| lines_written >= m) { break; }

        let src_off    = r * bpr;
        let row_data   = &data[src_off..src_off + bpr];
        let disp_off   = start_off.wrapping_add(src_off as u64);

        // Squeeze
        if opts.squeeze && row_data == prev_row.as_slice() {
            if !squeezed { out.write_all(b"*\n")?; squeezed = true; if opts.max_lines.is_some() { lines_written += 1; } }
            r += 1;
            continue;
        }
        if opts.squeeze { squeezed = false; prev_row.clear(); prev_row.extend_from_slice(row_data); }

        if let Some(core) = core {
            if pair_ok && r + 1 < full_rows {
                let row_len = core.prefix_len(field_len_for(opts, disp_off)) + core.full.emitted;
                let src_ptr = if src_off + 32 <= file_sz {
                    row_data.as_ptr()
                } else {
                    scratch[..32].copy_from_slice(&data[src_off..src_off + 32]);
                    scratch.as_ptr()
                };
                let n = unsafe { format_two_rows(row_buf.as_mut_ptr(), src_ptr, disp_off, core, &core.full, row_len) };
                out.write_all(&row_buf[..n])?;
                r += 2;
                continue;
            }
            let src_ptr = if src_off + core.blocks * 16 <= file_sz {
                row_data.as_ptr()
            } else {
                scratch[..bpr].copy_from_slice(row_data);
                scratch.as_ptr()
            };
            let n = unsafe { format_row(row_buf.as_mut_ptr(), src_ptr, disp_off, bpr, core, &core.full) };
            out.write_all(&row_buf[..n])?;
        } else {
            output_line(&mut out, row_data, disp_off, opts, do_color, hex_w, ascii_w)?;
        }
        if opts.max_lines.is_some() { lines_written += 1; }
        r += 1;
    }

    if tail_len > 0 && opts.max_lines.map_or(true, |m| lines_written < m) {
        let src_off  = full_rows * bpr;
        let disp_off = start_off.wrapping_add(src_off as u64);
        if let (Some(core), Some(cfg)) = (core, cfg.as_ref()) {
            scratch[..tail_len].copy_from_slice(&data[src_off..]);
            let layout = cfg.layout(tail_len);
            let n = unsafe { format_row(row_buf.as_mut_ptr(), scratch.as_ptr(), disp_off, tail_len, core, layout) };
            out.write_all(&row_buf[..n])?;
        } else {
            output_line(&mut out, &data[src_off..], disp_off, opts, do_color, hex_w, ascii_w)?;
        }
    }

    if opts.border != BorderStyle::None {
        border_bottom(&mut out, pos_w, hex_w, ascii_w, !opts.no_position,
                      has_ascii, opts.border)?;
    } else if !opts.no_position {
        // Final offset line (like xxd) — only printed if no border
        let final_off = start_off.wrapping_add(file_sz as u64);
        let mut tmp = [0u8; 20];
        let olen = write_offset(&mut tmp, final_off, opts.offset_dec, opts.uppercase);
        if do_color { out.write_all(ANSI_CYAN.as_bytes())?; }
        out.write_all(&tmp[..olen])?;
        if do_color { out.write_all(ANSI_RESET.as_bytes())?; }
        out.write_all(b"\n")?;
    }

    out.flush()
}

fn run_streaming(
    opts:      &Options,
    reader:    &mut dyn Read,
    do_color:  bool,
    use_simd:  bool,
    use_avx2:  bool,
) -> io::Result<()> {
    let bpr = opts.width;

    // Display offset = skip + jump
    let display_start: u64 = (opts.skip + opts.jump) as u64;

    let stdout = io::stdout();
    let mut out    = BufWriter::with_capacity(WRITE_BUF, stdout.lock());
    let mut rbuf   = vec![0u8; READ_BUF];
    let mut wbuf   = vec![0u8; WRITE_BUF + 128];
    let mut wpos   = 0usize;
    let mut offset = display_start;
    let mut total_read: u64 = 0;

    let pos_w = if opts.no_position { 0 } else {
        (if opts.offset_dec { 20 } else { offset_len(u64::MAX) }) + 1
    };
    let hex_w   = hex_width(opts);
    let ascii_w = ascii_width(opts);
    let has_ascii = !opts.no_ascii && !matches!(opts.mode, DisplayMode::Binary | DisplayMode::OneByteOctal | DisplayMode::TwoByteOctal | DisplayMode::OneByteDecimal | DisplayMode::TwoByteDecimal | DisplayMode::OneByteChar | DisplayMode::OneByteHex | DisplayMode::TwoByteHex);

    if opts.border != BorderStyle::None {
        border_top(&mut out, pos_w, hex_w, ascii_w, !opts.no_position, has_ascii, opts.border)?;
        border_header(&mut out, pos_w, hex_w, ascii_w, !opts.no_position, has_ascii, opts.border)?;
        border_sep(&mut out, pos_w, hex_w, ascii_w, !opts.no_position, has_ascii, opts.border)?;
    }

    let simd_eligible = is_simd_eligible(opts, do_color) && use_simd;
    let cfg = if simd_eligible {
        let kind = |tail: bool| {
            if opts.minimal {
                LayoutKind::Generic
            } else if do_color {
                LayoutKind::OutputLine
            } else if old_simd_eligible(opts) {
                if tail { LayoutKind::Scalar } else { LayoutKind::Generic }
            } else if opts.border != BorderStyle::None {
                LayoutKind::OutputLine
            } else {
                LayoutKind::Generic
            }
        };
        Some(RowCfg::new(opts, kind(false), kind(true)))
    } else { None };
    let core = cfg.as_ref().map(|c| &c.core);
    let mut row_tmp = vec![0u8; core.map_or(0, |c| c.blocks * 16).max(32)];
    let mut prev_row: Vec<u8> = Vec::new();
    let mut squeezed          = false;
    let mut lines_written: u64 = 0;

    // Partial-row carry buffer
    let mut carry     = vec![0u8; bpr];
    let mut carry_len = 0usize;

    loop {
        if opts.max_lines.map_or(false, |m| lines_written >= m) { break; }

        let max_read = match opts.length {
            Some(lim) => rbuf.len().min(lim.saturating_sub(total_read) as usize),
            None => rbuf.len(),
        };
        if max_read == 0 { break; }
        let n = reader.read(&mut rbuf[..max_read])?;
        if n == 0 { break; }
        total_read += n as u64;

        // Merge carry + new data
        let _data_start = 0usize;
        let _input: &[u8];

        // Fast path: combine carry + new bytes into a contiguous buffer
        let combined_len = carry_len + n;
        let full_rows = combined_len / bpr;
        let new_tail  = combined_len % bpr;

        // build a temp buffer only if there's a carry
        let scratch: Vec<u8>;
        let combined: &[u8] = if carry_len > 0 {
            scratch = {
                let mut v = Vec::with_capacity(combined_len);
                v.extend_from_slice(&carry[..carry_len]);
                v.extend_from_slice(&rbuf[..n]);
                v
            };
            &scratch
        } else {
            &rbuf[..n]
        };

        let pair_ok = simd_eligible && use_avx2 && !opts.squeeze
            && opts.max_lines.is_none() && can_pair(core.unwrap());
        let min_str = use_avx2 && opts.minimal && bpr == 16
            && !opts.no_position && !opts.offset_dec;
        let run_ok = (pair_ok && core.and_then(|c| c.full.fast.as_ref()).is_some() || min_str)
            && !opts.offset_dec
            && offset.wrapping_add(((full_rows.saturating_sub(1)) * bpr) as u64) <= 0xFFFF_FFFF;
        if run_ok {
            let core = core.unwrap();
            let lut = if opts.uppercase { HEX_UPPER } else { HEX_LOWER };
            let row_len = if opts.minimal {
                if opts.no_ascii { 42 } else { 59 }
            } else {
                core.prefix_len(8) + core.full.emitted
            };
            let mut r = 0usize;
            while r + 16 <= full_rows {
                if wpos + row_len * 16 + 16 > wbuf.len() { out.write_all(&wbuf[..wpos])?; wpos = 0; }
                if opts.minimal {
                    unsafe {
                        format_pairs_min_run(wbuf[wpos..].as_mut_ptr(), combined.as_ptr().add(r * bpr),
                                             offset.wrapping_add((r * bpr) as u64), 8, lut, !opts.no_ascii, row_len);
                    }
                } else {
                    let k = core.full.fast.as_ref().unwrap();
                    unsafe {
                        format_pairs_fast_run(wbuf[wpos..].as_mut_ptr(), combined.as_ptr().add(r * bpr),
                                              offset.wrapping_add((r * bpr) as u64), 8, lut, *k, row_len);
                    }
                }
                wpos += row_len * 16;
                r += 16;
            }
            let rem = full_rows - r;
            if rem >= 2 {
                if wpos + row_len * 16 + 16 > wbuf.len() { out.write_all(&wbuf[..wpos])?; wpos = 0; }
                if opts.minimal {
                    unsafe {
                        format_pairs_min_run(wbuf[wpos..].as_mut_ptr(), combined.as_ptr().add(r * bpr),
                                             offset.wrapping_add((r * bpr) as u64), rem / 2, lut, !opts.no_ascii, row_len);
                    }
                } else {
                    let k = core.full.fast.as_ref().unwrap();
                    unsafe {
                        format_pairs_fast_run(wbuf[wpos..].as_mut_ptr(), combined.as_ptr().add(r * bpr),
                                              offset.wrapping_add((r * bpr) as u64), rem / 2, lut, *k, row_len);
                    }
                }
                wpos += row_len * (rem / 2) * 2;
                r += (rem / 2) * 2;
            }
            if r < full_rows {
                let src = &combined[r * bpr..(r + 1) * bpr];
                let src_ptr = if r * bpr + core.blocks * 16 <= combined.len() {
                    src.as_ptr()
                } else {
                    row_tmp[..bpr].copy_from_slice(src);
                    row_tmp.as_ptr()
                };
                let n = unsafe { format_row(wbuf[wpos..].as_mut_ptr(), src_ptr, offset.wrapping_add((r * bpr) as u64), bpr, core, &core.full) };
                wpos += n;
            }
            offset = offset.wrapping_add((full_rows * bpr) as u64);
            carry_len = new_tail;
            if new_tail > 0 {
                carry[..new_tail].copy_from_slice(&combined[full_rows * bpr..]);
            }
            continue;
        }
        let mut r = 0usize;
        while r < full_rows {
            if opts.max_lines.map_or(false, |m| lines_written >= m) { break; }
            let src = &combined[r * bpr..(r + 1) * bpr];

            // Squeeze
            if opts.squeeze && src == prev_row.as_slice() {
                if !squeezed {
                    if wpos > 0 { out.write_all(&wbuf[..wpos])?; wpos = 0; }
                    out.write_all(b"*\n")?;
                    squeezed = true;
                }
                offset = offset.wrapping_add(bpr as u64);
                r += 1;
                continue;
            }
            if opts.squeeze { squeezed = false; prev_row.clear(); prev_row.extend_from_slice(src); }

            if simd_eligible {
                let core = core.unwrap();
                if pair_ok && r + 1 < full_rows {
                    let row_len = core.prefix_len(field_len_for(opts, offset)) + core.full.emitted;
                    if wpos + row_len * 2 + 16 > wbuf.len() { out.write_all(&wbuf[..wpos])?; wpos = 0; }
                    let src_ptr = if r * bpr + 32 <= combined.len() {
                        src.as_ptr()
                    } else {
                        row_tmp[..32].copy_from_slice(&combined[r * bpr..r * bpr + 32]);
                        row_tmp.as_ptr()
                    };
                    unsafe { format_two_rows(wbuf[wpos..].as_mut_ptr(), src_ptr, offset, core, &core.full, row_len); }
                    wpos += row_len * 2;
                    offset = offset.wrapping_add((2 * bpr) as u64);
                    r += 2;
                    continue;
                }
                let row_len = core.prefix_len(field_len_for(opts, offset)) + core.full.emitted;
                if wpos + row_len + 16 > wbuf.len() { out.write_all(&wbuf[..wpos])?; wpos = 0; }
                let src_ptr = if r * bpr + core.blocks * 16 <= combined.len() {
                    src.as_ptr()
                } else {
                    row_tmp[..bpr].copy_from_slice(src);
                    row_tmp.as_ptr()
                };
                let n = unsafe { format_row(wbuf[wpos..].as_mut_ptr(), src_ptr, offset, bpr, core, &core.full) };
                wpos += n;
            } else if do_color {
                if wpos > 0 { out.write_all(&wbuf[..wpos])?; wpos = 0; }
                output_line(&mut out, src, offset, opts, do_color, hex_w, ascii_w)?;
            } else {
                let mut tmp = Vec::with_capacity(128);
                format_row_generic(&mut tmp, src, offset, opts);
                if wpos + tmp.len() > wbuf.len() { out.write_all(&wbuf[..wpos])?; wpos = 0; }
                wbuf[wpos..wpos+tmp.len()].copy_from_slice(&tmp);
                wpos += tmp.len();
            }
            if opts.max_lines.is_some() { lines_written += 1; }
            offset = offset.wrapping_add(bpr as u64);
            r += 1;
        }

        // Save tail into carry
        carry_len = new_tail;
        if new_tail > 0 {
            carry[..new_tail].copy_from_slice(&combined[full_rows * bpr..]);
        }
    }

    // Flush remaining carry (partial row)
    if carry_len > 0 && opts.max_lines.map_or(true, |m| lines_written < m) {
        let src = &carry[..carry_len];
        if simd_eligible {
            let core = core.unwrap();
            let row_len = core.prefix_len(field_len_for(opts, offset)) + core.full.emitted;
            if wpos + row_len + 16 > wbuf.len() { out.write_all(&wbuf[..wpos])?; wpos = 0; }
            row_tmp[..carry_len].copy_from_slice(src);
            let layout = cfg.as_ref().unwrap().layout(carry_len);
            let n = unsafe { format_row(wbuf[wpos..].as_mut_ptr(), row_tmp.as_ptr(), offset, carry_len, core, layout) };
            wpos += n;
        } else if do_color {
            if wpos > 0 { out.write_all(&wbuf[..wpos])?; wpos = 0; }
            output_line(&mut out, src, offset, opts, do_color, hex_w, ascii_w)?;
        } else {
            let mut tmp = Vec::with_capacity(128);
            format_row_generic(&mut tmp, src, offset, opts);
            if wpos + tmp.len() > wbuf.len() { out.write_all(&wbuf[..wpos])?; wpos = 0; }
            wbuf[wpos..wpos+tmp.len()].copy_from_slice(&tmp);
            wpos += tmp.len();
        }
        offset = offset.wrapping_add(carry_len as u64);
    }

    if wpos > 0 { out.write_all(&wbuf[..wpos])?; }

    // Final offset line
    {
        let mut tmp = [0u8; 20];
        let olen = write_offset(&mut tmp, offset, opts.offset_dec, opts.uppercase);
        if do_color { out.write_all(ANSI_CYAN.as_bytes())?; }
        out.write_all(&tmp[..olen])?;
        if do_color { out.write_all(ANSI_RESET.as_bytes())?; }
        out.write_all(b"\n")?;
    }

    if opts.squeeze {
        // Already handled inline
    }

    if opts.border != BorderStyle::None {
        border_bottom(&mut out, pos_w, hex_w, ascii_w, !opts.no_position, has_ascii, opts.border)?;
    }

    out.flush()
}

