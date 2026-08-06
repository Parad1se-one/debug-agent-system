# TOOL-DMP dmp_parser

Safe DMP evidence parser. It reads only bounded header bytes from `.dmp` / `.mdmp`
files, reports signature/size/hash/dump-kind hints, and marks the file as
WinDbg-ready. It does not execute WinDbg, scan full memory, extract strings from
the full dump, or infer root cause.
