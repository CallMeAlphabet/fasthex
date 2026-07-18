//! fasthex 0.3.0 – a very fast hex dumper
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
use std::arch::x86_64::*;
use std::env;
use std::fs::File;
use std::io::{self, BufWriter, IsTerminal, Read, Seek, SeekFrom, Write};
use std::sync::mpsc::{channel, sync_channel};
use std::thread;

const READ_BUF: usize = 256 * 1024;
const WRITE_BUF: usize = 4 * 1024 * 1024;
const PIPE_SIZE_HINT: libc::c_int = 2 * 1024 * 1024;
const _CHUNK_ROWS: usize = (64 * 1024 * 1024) / 76; // recalculated per-mode at runtime

static HEX_LOWER: &[u8; 16] = b"0123456789abcdef";
static HEX_UPPER: &[u8; 16] = b"0123456789ABCDEF";

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

struct Options {
    mode:         DisplayMode,
    width:        usize,        // bytes per row (0 = auto for unicode border)
    group:        usize,        // bytes per group: 1,2,4,8
    endian:       Endian,
    border:       BorderStyle,
    no_ascii:     bool,
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
    let mut page = HelpPage::new("fasthex 0.3.0 - a very fast hex dumper")
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
                "version" => { println!("0.3.0"); std::process::exit(0); }
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
                    b'v' => { println!("0.3.0"); std::process::exit(0); }
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

#[inline(always)]
unsafe fn write_offset_n(dst: *mut u8, off: u64, n: usize) {
    for k in 0..n {
        *dst.add(n - 1 - k) = HEX_LOWER[((off >> (k * 4)) & 0xf) as usize];
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
        let mut fds = [0i32; 2];
        if unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
            return Err(io::Error::last_os_error());
        }
        unsafe { libc::fcntl(fds[1], libc::F_SETPIPE_SZ, PIPE_SIZE_HINT); }
        Ok(Self { pipe_r: fds[0], pipe_w: fds[1], stdout: libc::STDOUT_FILENO, fallback: false })
    }

    fn write_chunk(&mut self, buf: &[u8]) -> io::Result<()> {
        if self.fallback { return self.write_fallback(buf); }
        unsafe { self.write_zero_copy(buf) }
    }

    unsafe fn write_zero_copy(&mut self, buf: &[u8]) -> io::Result<()> {
        let mut vsrc = buf.as_ptr();
        let mut vremain = buf.len();
        while vremain > 0 {
            let iov = libc::iovec { iov_base: vsrc as *mut _, iov_len: vremain };
            let vspliced = libc::vmsplice(self.pipe_w, &iov, 1, libc::SPLICE_F_GIFT);
            if vspliced < 0 { self.drain_pipe_fallback(buf)?; return Ok(()); }
            let vspliced = vspliced as usize;
            vsrc = vsrc.add(vspliced);
            vremain -= vspliced;
            let mut sremain = vspliced;
            while sremain > 0 {
                let n = libc::splice(self.pipe_r, std::ptr::null_mut(),
                                     self.stdout, std::ptr::null_mut(),
                                     sremain, libc::SPLICE_F_MOVE);
                if n < 0 {
                    let e = io::Error::last_os_error();
                    if e.raw_os_error() == Some(libc::EINVAL) {
                        self.drain_pipe_fallback(buf)?;
                        return Ok(());
                    }
                    return Err(e);
                }
                sremain -= n as usize;
            }
        }
        Ok(())
    }

    fn drain_pipe_fallback(&mut self, full: &[u8]) -> io::Result<()> {
        self.fallback = true;
        let mut tmp = vec![0u8; 65536];
        loop {
            let n = unsafe { libc::read(self.pipe_r, tmp.as_mut_ptr() as *mut _, tmp.len()) };
            if n <= 0 { break; }
            io::stdout().lock().write_all(&tmp[..n as usize])?;
        }
        self.write_fallback(full)
    }

    fn write_fallback(&self, buf: &[u8]) -> io::Result<()> {
        io::stdout().lock().write_all(buf)
    }
}

impl Drop for ZeroCopyWriter {
    fn drop(&mut self) {
        unsafe { libc::close(self.pipe_r); libc::close(self.pipe_w); }
    }
}

macro_rules! expand_and_store {
    ($dst:expr, $pairs_lo:expr, $pairs_hi:expr, $ascii:expr) => {{
        let p = $dst;
        let spaces = _mm_set1_epi8(b' ' as i8);
        let zero   = _mm_setzero_si128();
        let shuf_a = _mm_setr_epi8(0,1,-1,2,3,-1,4,5,-1,6,7,-1,-1,-1,-1,-1);
        let shuf_b = _mm_setr_epi8(8,9,-1,10,11,-1,12,13,-1,14,15,-1,-1,-1,-1,-1);
        macro_rules! expand {
            ($pairs:expr, $shuf:expr) => {{
                let c = _mm_shuffle_epi8($pairs, $shuf);
                _mm_blendv_epi8(c, spaces, _mm_cmpeq_epi8(c, zero))
            }};
        }
        let c1 = expand!($pairs_lo, shuf_a);
        let c2 = expand!($pairs_lo, shuf_b);
        let c3 = expand!($pairs_hi, shuf_a);
        let c4 = expand!($pairs_hi, shuf_b);
        _mm_storeu_si128(p.add(9)  as *mut __m128i, c1);
        _mm_storeu_si128(p.add(21) as *mut __m128i, c2);
        *p.add(33) = b' ';
        _mm_storeu_si128(p.add(34) as *mut __m128i, c3);
        _mm_storeu_si128(p.add(46) as *mut __m128i, c4);
        *p.add(58) = b' ';
        *p.add(59) = b'|';
        _mm_storeu_si128(p.add(60) as *mut __m128i, $ascii);
        *p.add(76) = b'|';
        *p.add(77) = b'\n';
    }};
}

#[target_feature(enable = "avx2,ssse3,sse4.1")]
unsafe fn format_two_rows_avx2(dst: *mut u8, src: *const u8, off: u64, off_len: usize) {
    let orb = off_len + 71; // +2 for | | around ASCII panel
    write_offset_n(dst, off, off_len);
    *dst.add(off_len) = b':';
    *dst.add(off_len + 1) = b' ';
    write_offset_n(dst.add(orb), off.wrapping_add(16), off_len);
    *dst.add(orb + off_len) = b':';
    *dst.add(orb + off_len + 1) = b' ';

    let input  = _mm256_loadu_si256(src as *const __m256i);
    let lo_msk = _mm256_set1_epi8(0x0f_u8 as i8);
    let lut    = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        b'0' as i8,b'1' as i8,b'2' as i8,b'3' as i8,b'4' as i8,b'5' as i8,
        b'6' as i8,b'7' as i8,b'8' as i8,b'9' as i8,b'a' as i8,b'b' as i8,
        b'c' as i8,b'd' as i8,b'e' as i8,b'f' as i8));
    let lo  = _mm256_and_si256(input, lo_msk);
    let hi  = _mm256_and_si256(_mm256_srli_epi16(input, 4), lo_msk);
    let hlo = _mm256_shuffle_epi8(lut, lo);
    let hhi = _mm256_shuffle_epi8(lut, hi);
    let plo = _mm256_unpacklo_epi8(hhi, hlo);
    let phi = _mm256_unpackhi_epi8(hhi, hlo);

    let r0_plo = _mm256_castsi256_si128(plo);
    let r0_phi = _mm256_castsi256_si128(phi);
    let r1_plo = _mm256_extracti128_si256(plo, 1);
    let r1_phi = _mm256_extracti128_si256(phi, 1);

    let dot  = _mm256_set1_epi8(b'.' as i8);
    let low  = _mm256_set1_epi8(0x1f_u8 as i8);
    let high = _mm256_set1_epi8(0x7f_u8 as i8);
    let pr   = _mm256_and_si256(
        _mm256_cmpgt_epi8(input, low), _mm256_cmpgt_epi8(high, input));
    let asc  = _mm256_blendv_epi8(dot, input, pr);
    let asc0 = _mm256_castsi256_si128(asc);
    let asc1 = _mm256_extracti128_si256(asc, 1);

    expand_and_store!(dst.add(off_len - 7), r0_plo, r0_phi, asc0);
    expand_and_store!(dst.add(orb + off_len - 7), r1_plo, r1_phi, asc1);
}

#[target_feature(enable = "ssse3,sse4.1")]
unsafe fn format_row_simd(dst: *mut u8, src: *const u8, off: u64, off_len: usize) {
    write_offset_n(dst, off, off_len);
    *dst.add(off_len) = b':';
    *dst.add(off_len + 1) = b' ';
    // Shift base so expand_and_store!(base) writes hex at base[9] and newline at base[75].
    // off_len=8: base=dst,   row=76 bytes. off_len=9: base=dst+1, row=77 bytes. All SIMD.
    let dst = dst.add(off_len - 7);

    let input  = _mm_loadu_si128(src as *const __m128i);
    let lo_msk = _mm_set1_epi8(0x0f_u8 as i8);
    let lut    = _mm_setr_epi8(
        b'0' as i8,b'1' as i8,b'2' as i8,b'3' as i8,b'4' as i8,b'5' as i8,
        b'6' as i8,b'7' as i8,b'8' as i8,b'9' as i8,b'a' as i8,b'b' as i8,
        b'c' as i8,b'd' as i8,b'e' as i8,b'f' as i8);
    let lo  = _mm_and_si128(input, lo_msk);
    let hi  = _mm_and_si128(_mm_srli_epi16(input, 4), lo_msk);
    let hlo = _mm_shuffle_epi8(lut, lo);
    let hhi = _mm_shuffle_epi8(lut, hi);
    let plo = _mm_unpacklo_epi8(hhi, hlo);
    let phi = _mm_unpackhi_epi8(hhi, hlo);

    let pr  = _mm_and_si128(
        _mm_cmpgt_epi8(input, _mm_set1_epi8(0x1f_u8 as i8)),
        _mm_cmpgt_epi8(_mm_set1_epi8(0x7f_u8 as i8), input));
    let asc = _mm_blendv_epi8(_mm_set1_epi8(b'.' as i8), input, pr);

    expand_and_store!(dst, plo, phi, asc);
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
    opts.mode == DisplayMode::Canonical
        && opts.width == 16
        && opts.group == 1
        && opts.endian == Endian::Big
        && !opts.no_ascii
        && !opts.no_position
        && opts.border == BorderStyle::None
        && !do_color
        && !opts.uppercase
        && !opts.offset_dec
        && opts.table == CharTable::Ascii
        && opts.formats.is_empty()
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
                    let mut slice: &[u8] = data;
                    return run_plain(&opts, &mut slice);
                }
                DisplayMode::CInclude => {
                    let mut slice: &[u8] = data;
                    return run_c_include(&opts, &mut slice);
                }
                _ => {}
            }
            if !opts.formats.is_empty() {
                return run_custom_format(&opts, data, start_disp);
            }

            let simd_ok = is_simd_eligible(&opts, do_color);
            let use_avx2 = simd_ok && is_x86_feature_detected!("avx2");
            let use_simd = simd_ok && (use_avx2 ||
                (is_x86_feature_detected!("ssse3") && is_x86_feature_detected!("sse4.1")));

            let needs_serial = opts.mode != DisplayMode::Canonical
                || do_color || opts.squeeze || opts.border != BorderStyle::None
                || opts.max_lines.is_some() || !opts.formats.is_empty()
                || !use_simd; // scalar parallel path ignores opts entirely; serial handles all non-SIMD cases

            if needs_serial {
                return run_serial_mmap(&opts, data, start_disp, do_color, use_simd);
            } else {
                return run_parallel_mmap(&opts, data, start_disp, use_avx2, use_simd);
            }
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
    let use_simd  = simd_ok &&
        is_x86_feature_detected!("ssse3") && is_x86_feature_detected!("sse4.1");

    run_streaming(&opts, &mut reader, do_color, use_simd)
}

fn run_parallel_mmap(
    opts:     &Options,
    data:     &[u8],
    start_off: u64,
    use_avx2: bool,
    use_simd: bool,
) -> io::Result<()> {
    let bpr     = opts.width;
    let max_off       = start_off.saturating_add(data.len() as u64);
    let off_len_simd  = offset_len(max_off);
    let orb: usize = if use_simd && bpr == 16 {
        off_len_simd + 71
    } else {
        // scalar parallel: offset(up to 20) + ": " + bpr*3 hex + half-space + "|" + bpr ascii + "|\n"
        20 + 2 + bpr * 3 + 1 + 2 + bpr + 1
    };
    let file_sz   = data.len();
    let full_rows = file_sz / bpr;
    let tail_len  = file_sz % bpr;
    let chunk_rows = (64 * 1024 * 1024) / orb;
    let buf_cap    = chunk_rows * orb;

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

    let mut row_cursor = 0usize;
    while row_cursor < full_rows {
        let rows = (full_rows - row_cursor).min(chunk_rows);

        #[cfg(unix)]
        {
            let pf_start = (row_cursor + 2 * chunk_rows) * bpr;
            if pf_start < file_sz {
                let pf_len = (chunk_rows * bpr).min(file_sz - pf_start);
                unsafe {
                    libc::madvise(data.as_ptr().add(pf_start) as *mut libc::c_void,
                                  pf_len, libc::MADV_WILLNEED);
                }
            }
        }

        let mut chunk_out = recv_free.recv().unwrap();
        chunk_out.resize(rows * orb, 0);

        if use_avx2 {
            let even = rows & !1;
            chunk_out[..even * orb]
                .par_chunks_mut(orb * 2)
                .enumerate()
                .for_each(|(i, two_rows)| {
                    let src_off = (row_cursor + i * 2) * bpr;
                    let off = start_off.wrapping_add(src_off as u64);
                    unsafe {
                        format_two_rows_avx2(two_rows.as_mut_ptr(), data.as_ptr().add(src_off), off, off_len_simd);
                    }
                });
            if rows & 1 != 0 {
                let src_off = (row_cursor + rows - 1) * bpr;
                let off = start_off.wrapping_add(src_off as u64);
                unsafe {
                    format_row_simd(chunk_out[(rows-1)*orb..].as_mut_ptr(),
                                    data.as_ptr().add(src_off), off, off_len_simd);
                }
            }
        } else if use_simd {
            chunk_out
                .par_chunks_mut(orb)
                .enumerate()
                .for_each(|(i, row)| {
                    let src_off = (row_cursor + i) * bpr;
                    let off = start_off.wrapping_add(src_off as u64);
                    unsafe {
                        format_row_simd(row.as_mut_ptr(), data.as_ptr().add(src_off), off, off_len_simd);
                    }
                });
        } else {
            // scalar parallel (non-16-width or no SIMD)
            chunk_out
                .par_chunks_mut(orb)
                .enumerate()
                .for_each(|(i, row)| {
                    let src_off = (row_cursor + i) * bpr;
                    let off = start_off.wrapping_add(src_off as u64);
                    let mut v = Vec::with_capacity(orb);
                    format_canonical_scalar(&mut v, &data[src_off..src_off+bpr], off);
                    row[..v.len().min(orb)].copy_from_slice(&v[..v.len().min(orb)]);
                });
        }

        send_data.send(chunk_out).unwrap();
        row_cursor += rows;
    }

    drop(send_data);
    writer.join().unwrap()?;

    // Tail row (partial) must come before the final offset line
    if tail_len > 0 {
        let src_off = full_rows * bpr;
        let off     = start_off.wrapping_add(src_off as u64);
        let mut v   = Vec::with_capacity(orb);
        format_canonical_scalar(&mut v, &data[src_off..], off);
        io::stdout().lock().write_all(&v)?;
    }

    // Final offset line (like xxd: always printed at the end)
    {
        let final_off = start_off.wrapping_add(file_sz as u64);
        let mut tmp   = [0u8; 20];
        let olen      = write_offset(&mut tmp, final_off, false, false);
        io::stdout().lock().write_all(&tmp[..olen])?;
        io::stdout().lock().write_all(b"\n")?;
    }

    Ok(())
}

/// Minimal canonical scalar formatter for the parallel path (no opts dependency).
fn format_canonical_scalar(dst: &mut Vec<u8>, src: &[u8], off: u64) {
    let n     = src.len();
    let width = 16; // Always 16 for canonical
    let half  = width / 2;
    let mut tmp = [0u8; 20]; // enough for up to 16 hex digits
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
            dst.push(b' ');
        } else { dst.push(b' '); dst.push(b' '); dst.push(b' '); }
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

fn run_serial_mmap(
    opts:      &Options,
    data:      &[u8],
    start_off: u64,
    do_color:  bool,
    use_simd:  bool,
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

    for r in 0..full_rows {
        if opts.max_lines.map_or(false, |m| lines_written >= m) { break; }

        let src_off    = r * bpr;
        let row_data   = &data[src_off..src_off + bpr];
        let disp_off   = start_off.wrapping_add(src_off as u64);

        // Squeeze
        if opts.squeeze && row_data == prev_row.as_slice() {
            if !squeezed { out.write_all(b"*\n")?; squeezed = true; if opts.max_lines.is_some() { lines_written += 1; } }
            continue;
        }
        if opts.squeeze { squeezed = false; prev_row.clear(); prev_row.extend_from_slice(row_data); }

        // Fast SIMD path for canonical no-color
        if simd_ok && use_simd && opts.border == BorderStyle::None {
            let off_len   = offset_len(disp_off);
            let row_bytes = off_len + 71; // 78 for <4GiB, 79 for >=4GiB
            let mut row   = [0u8; 88];   // 16 hex digits + 71 = 87 max possible
            unsafe { format_row_simd(row.as_mut_ptr(), row_data.as_ptr(), disp_off, off_len); }
            out.write_all(&row[..row_bytes])?;
        } else {
            output_line(&mut out, row_data, disp_off, opts, do_color, hex_w, ascii_w)?;
        }
        if opts.max_lines.is_some() { lines_written += 1; }
    }

    if tail_len > 0 && opts.max_lines.map_or(true, |m| lines_written < m) {
        let src_off  = full_rows * bpr;
        let disp_off = start_off.wrapping_add(src_off as u64);
        output_line(&mut out, &data[src_off..], disp_off, opts, do_color, hex_w, ascii_w)?;
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

        for r in 0..full_rows {
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
                continue;
            }
            if opts.squeeze { squeezed = false; prev_row.clear(); prev_row.extend_from_slice(src); }

            if simd_eligible {
                let off_len   = offset_len(offset);
                let row_bytes = off_len + 71;
                if wpos + row_bytes > wbuf.len() { out.write_all(&wbuf[..wpos])?; wpos = 0; }
                unsafe { format_row_simd(wbuf[wpos..].as_mut_ptr(), src.as_ptr(), offset, off_len); }
                wpos += row_bytes;
            } else if do_color || opts.border != BorderStyle::None {
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
            // partial row: use scalar
            let mut tmp = Vec::with_capacity(128);
            format_canonical_scalar(&mut tmp, src, offset);
            if wpos + tmp.len() > wbuf.len() { out.write_all(&wbuf[..wpos])?; wpos = 0; }
            wbuf[wpos..wpos+tmp.len()].copy_from_slice(&tmp);
            wpos += tmp.len();
        } else if do_color || opts.border != BorderStyle::None {
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




