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
//!
//! Row layout – 76 bytes:
//!   [0..7]   8-digit hex offset
//!   [8]      ' '
//!   [9..32]  hex group-1  (8 bytes × "XX ")
//!   [33]     extra gap space
//!   [34..57] hex group-2  (8 bytes × "XX ")
//!   [58]     ' '
//!   [59..74] ASCII printable / '.'
//!   [75]     '\n'
//!
//! NOTE: the offset field is u32, so files larger than 4 GiB will have a
//! wrapping offset display. Known limitation.

#![allow(clippy::missing_safety_doc)]

use memmap2::Mmap;
use rayon::prelude::*;
use std::arch::x86_64::*;
use std::env;
use std::fs::File;
use std::io::{self, Read, Write};
use std::sync::mpsc::{channel, sync_channel};
use std::thread;

const READ_BUF:       usize          = 256 * 1024;
const WRITE_BUF:      usize          = 4 * 1024 * 1024;
const ROW_BYTES:      usize          = 76;
const CHUNK_ROWS:     usize          = (64 * 1024 * 1024) / 16;
const PIPE_SIZE_HINT: libc::c_int   = 2 * 1024 * 1024;

static HEX: &[u8; 16] = b"0123456789abcdef";

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
        let mut vsrc     = buf.as_ptr();
        let mut vremain  = buf.len();

        while vremain > 0 {
            let iov = libc::iovec {
                iov_base: vsrc as *mut libc::c_void,
                iov_len:  vremain,
            };
            let vspliced = libc::vmsplice(
                self.pipe_w,
                &iov,
                1,
                libc::SPLICE_F_GIFT,
            );
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
                libc::read(self.pipe_r, tmp.as_mut_ptr() as *mut libc::c_void, tmp.len())
            };
            if n <= 0 { break; }
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
    *dst.add(5) = HEX[((off >>  8) & 0xf) as usize];
    *dst.add(6) = HEX[((off >>  4) & 0xf) as usize];
    *dst.add(7) = HEX[( off        & 0xf) as usize];
}

macro_rules! expand_and_store {
    ($dst:expr, $pairs_lo:expr, $pairs_hi:expr, $ascii:expr) => {{
        let dst_base: *mut u8 = $dst;

        let spaces = _mm_set1_epi8(b' ' as i8);
        let zero   = _mm_setzero_si128();

        let shuf_a = _mm_setr_epi8(0,1,-1, 2,3,-1, 4,5,-1, 6,7,-1, -1,-1,-1,-1);
        let shuf_b = _mm_setr_epi8(8,9,-1, 10,11,-1, 12,13,-1, 14,15,-1, -1,-1,-1,-1);

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
        _mm_storeu_si128(p.add( 0) as *mut __m128i, chunk1);
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
    write_offset(dst.add(ROW_BYTES), off.wrapping_add(16));
    *dst.add(ROW_BYTES + 8) = b' ';

    let input   = _mm256_loadu_si256(src as *const __m256i);
    let lo_mask = _mm256_set1_epi8(0x0f_u8 as i8);
    let lut     = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        b'0' as i8, b'1' as i8, b'2' as i8, b'3' as i8,
        b'4' as i8, b'5' as i8, b'6' as i8, b'7' as i8,
        b'8' as i8, b'9' as i8, b'a' as i8, b'b' as i8,
        b'c' as i8, b'd' as i8, b'e' as i8, b'f' as i8,
    ));

    let lo_nib   = _mm256_and_si256(input, lo_mask);
    let hi_nib   = _mm256_and_si256(_mm256_srli_epi16(input, 4), lo_mask);
    let hex_lo   = _mm256_shuffle_epi8(lut, lo_nib);
    let hex_hi   = _mm256_shuffle_epi8(lut, hi_nib);
    let pairs_lo = _mm256_unpacklo_epi8(hex_hi, hex_lo);
    let pairs_hi = _mm256_unpackhi_epi8(hex_hi, hex_lo);

    let r0_plo = _mm256_castsi256_si128(pairs_lo);
    let r0_phi = _mm256_castsi256_si128(pairs_hi);
    let r1_plo = _mm256_extracti128_si256(pairs_lo, 1);
    let r1_phi = _mm256_extracti128_si256(pairs_hi, 1);

    let dot       = _mm256_set1_epi8(b'.' as i8);
    let low       = _mm256_set1_epi8(0x1f_u8 as i8);
    let high      = _mm256_set1_epi8(0x7f_u8 as i8);
    let printable = _mm256_and_si256(
        _mm256_cmpgt_epi8(input, low),
        _mm256_cmpgt_epi8(high,  input),
    );
    let ascii = _mm256_blendv_epi8(dot, input, printable);

    let ascii_r0 = _mm256_castsi256_si128(ascii);
    let ascii_r1 = _mm256_extracti128_si256(ascii, 1);

    expand_and_store!(dst,                r0_plo, r0_phi, ascii_r0);
    expand_and_store!(dst.add(ROW_BYTES), r1_plo, r1_phi, ascii_r1);
}

#[target_feature(enable = "ssse3,sse4.1")]
unsafe fn format_row_simd(dst: *mut u8, src: *const u8, off: u32) {
    write_offset(dst, off);
    *dst.add(8) = b' ';

    let input   = _mm_loadu_si128(src as *const __m128i);
    let lo_mask = _mm_set1_epi8(0x0f_u8 as i8);
    let lut     = _mm_setr_epi8(
        b'0' as i8, b'1' as i8, b'2' as i8, b'3' as i8,
        b'4' as i8, b'5' as i8, b'6' as i8, b'7' as i8,
        b'8' as i8, b'9' as i8, b'a' as i8, b'b' as i8,
        b'c' as i8, b'd' as i8, b'e' as i8, b'f' as i8,
    );

    let lo_nib   = _mm_and_si128(input, lo_mask);
    let hi_nib   = _mm_and_si128(_mm_srli_epi16(input, 4), lo_mask);
    let hex_lo   = _mm_shuffle_epi8(lut, lo_nib);
    let hex_hi   = _mm_shuffle_epi8(lut, hi_nib);
    let pairs_lo = _mm_unpacklo_epi8(hex_hi, hex_lo);
    let pairs_hi = _mm_unpackhi_epi8(hex_hi, hex_lo);

    let printable = _mm_and_si128(
        _mm_cmpgt_epi8(input, _mm_set1_epi8(0x1f_u8 as i8)),
        _mm_cmpgt_epi8(_mm_set1_epi8(0x7f_u8 as i8), input),
    );
    let ascii = _mm_blendv_epi8(_mm_set1_epi8(b'.' as i8), input, printable);

    expand_and_store!(dst, pairs_lo, pairs_hi, ascii);
}

fn format_row_partial(dst: &mut [u8], src: &[u8], off: u32) {
    let n = src.len();
    unsafe { write_offset(dst.as_mut_ptr(), off) };
    dst[8] = b' ';
    dst[9..59].fill(b' ');

    for i in 0..n {
        let pos = if i < 8 { i * 3 } else { 25 + (i - 8) * 3 };
        dst[9 + pos]     = HEX[(src[i] >> 4) as usize];
        dst[9 + pos + 1] = HEX[(src[i] & 0xf) as usize];
    }

    for i in 0..16 {
        dst[59 + i] = if i < n {
            let c = src[i];
            if c >= 0x20 && c <= 0x7e { c } else { b'.' }
        } else {
            b' '
        };
    }
    dst[75] = b'\n';
}

fn main() -> io::Result<()> {
    let use_avx2 = is_x86_feature_detected!("avx2");
    let use_simd = use_avx2
        || (is_x86_feature_detected!("ssse3") && is_x86_feature_detected!("sse4.1"));

    let file_opt: Option<File> = match env::args_os().nth(1) {
        Some(path) => Some(File::open(path)?),
        None       => None,
    };

    if let Some(ref file) = file_opt {
        if let Ok(mmap) = unsafe { Mmap::map(file) } {
            #[cfg(unix)]
            unsafe {
                libc::madvise(
                    mmap.as_ptr() as *mut libc::c_void,
                    mmap.len(),
                    libc::MADV_SEQUENTIAL,
                );
            }

            let data      = mmap.as_ref();
            let file_size = data.len();
            let full_rows = file_size / 16;
            let tail_len  = file_size % 16;

            let buf_cap = CHUNK_ROWS * ROW_BYTES;

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
                let rows = (full_rows - row_cursor).min(CHUNK_ROWS);

                #[cfg(unix)]
                {
                    let pf_start = (row_cursor + 2 * CHUNK_ROWS) * 16;
                    if pf_start < file_size {
                        let pf_len = (CHUNK_ROWS * 16).min(file_size - pf_start);
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
                chunk_out.resize(rows * ROW_BYTES, 0);

                if use_avx2 {
                    let even = rows & !1;
                    chunk_out[..even * ROW_BYTES]
                        .par_chunks_mut(ROW_BYTES * 2)
                        .enumerate()
                        .for_each(|(i, two_rows)| {
                            let src_off = (row_cursor + i * 2) * 16;
                            unsafe {
                                format_two_rows_avx2(
                                    two_rows.as_mut_ptr(),
                                    data.as_ptr().add(src_off),
                                    src_off as u32,
                                );
                            }
                        });
                    if rows & 1 != 0 {
                        let src_off = (row_cursor + rows - 1) * 16;
                        unsafe {
                            format_row_simd(
                                chunk_out[(rows - 1) * ROW_BYTES..].as_mut_ptr(),
                                data.as_ptr().add(src_off),
                                src_off as u32,
                            );
                        }
                    }
                } else if use_simd {
                    chunk_out
                        .par_chunks_mut(ROW_BYTES)
                        .enumerate()
                        .for_each(|(i, row)| {
                            let src_off = (row_cursor + i) * 16;
                            unsafe {
                                format_row_simd(
                                    row.as_mut_ptr(),
                                    data.as_ptr().add(src_off),
                                    src_off as u32,
                                );
                            }
                        });
                } else {
                    chunk_out
                        .par_chunks_mut(ROW_BYTES)
                        .enumerate()
                        .for_each(|(i, row)| {
                            let src_off = (row_cursor + i) * 16;
                            format_row_partial(row, &data[src_off..src_off + 16], src_off as u32);
                        });
                }

                send_data.send(chunk_out).unwrap();
                row_cursor += rows;
            }

            drop(send_data);
            writer.join().unwrap()?;

            if tail_len > 0 {
                let src_off = full_rows * 16;
                let mut row = vec![0u8; ROW_BYTES];
                format_row_partial(&mut row, &data[src_off..], src_off as u32);
                io::stdout().lock().write_all(&row)?;
            }

            return Ok(());
        }
    }

    let mut reader: Box<dyn Read> = match file_opt {
        Some(f) => Box::new(f),
        None    => Box::new(io::stdin()),
    };

    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut rbuf   = vec![0u8; READ_BUF];
    let mut wbuf   = vec![0u8; WRITE_BUF + ROW_BYTES];
    let mut wpos   = 0usize;
    let mut offset = 0u32;

    loop {
        let n = reader.read(&mut rbuf)?;
        if n == 0 { break; }

        let full = n / 16;
        let tail = n % 16;

        for r in 0..full {
            if wpos > WRITE_BUF - ROW_BYTES {
                out.write_all(&wbuf[..wpos])?;
                wpos = 0;
            }
            if use_simd {
                unsafe {
                    format_row_simd(
                        wbuf[wpos..].as_mut_ptr(),
                        rbuf[r * 16..].as_ptr(),
                        offset,
                    );
                }
            } else {
                format_row_partial(
                    &mut wbuf[wpos..wpos + ROW_BYTES],
                    &rbuf[r * 16..r * 16 + 16],
                    offset,
                );
            }
            wpos   += ROW_BYTES;
            offset += 16;
        }

        if tail > 0 {
            if wpos > WRITE_BUF - ROW_BYTES {
                out.write_all(&wbuf[..wpos])?;
                wpos = 0;
            }
            let base = full * 16;
            format_row_partial(
                &mut wbuf[wpos..wpos + ROW_BYTES],
                &rbuf[base..base + tail],
                offset,
            );
            wpos   += ROW_BYTES;
            offset += tail as u32;
        }
    }

    if wpos > 0 {
        out.write_all(&wbuf[..wpos])?;
    }

    Ok(())
}
