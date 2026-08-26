# Security Policy

## Supported Versions

Only the latest released version of this tool is supported.

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Report via:
- Discord: `itzalphabet`
- GitHub: [@CallMeAlphabet](https://github.com/CallMeAlphabet)
- Email: callmeletters@gmail.com

I'll respond as quickly as I can. For valid vulnerabilities I'll issue a
fix and credit you in the release notes if you'd like.

## Scope

These are CLI tools that run with your own user permissions on your own
data. The attack surface is narrow, but things worth reporting include:

- Unsafe memory handling (buffer overflows, use-after-free)
- Incorrect output that could cause data loss in a pipeline
- SIMD paths producing different results than the scalar fallback
