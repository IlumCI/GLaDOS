"""Extract RTL8188EU register and descriptor constants from the Linux source.

Provenance, same rule as the initialisation tables this appends to: the values
are Linux's, from the GPL-2.0 rtl8xxxu driver, and they are *converted* rather
than retyped. Nothing here is written from memory.

Run it against a checkout or a download of the four files it names; it prints
every value it extracted so the output can be read against the source, and it
fails rather than guessing if anything it was asked for is missing, duplicated
after preprocessing, or in a form it does not understand.

Two traps this exists to avoid, both found the hard way:

  * `TXDESC_OWN` is defined twice in rtl8xxxu.h with different values. The
    first is inside `#if 0`. A grep takes the dead one, which is BIT(31)
    against a live BIT(7), and the descriptor's OWN bit lands in the wrong
    byte. So the dead branches are stripped before anything is matched.

  * The RX descriptor is a C bitfield, and its field offsets exist nowhere as
    numbers -- they are implied by declaration order under __LITTLE_ENDIAN,
    where the first member takes the least significant bits. They are computed
    from the declaration here rather than counted by hand.

usage: rtlconv.py <dir-with-sources> [--emit path/to/rtl8188eu_tables.rs]
"""

import hashlib
import os
import re
import sys


def strip_dead(text):
    """Remove `#if 0` branches, keeping the `#else` side.

    Deliberately not a C preprocessor. It handles the one construct that
    appears in these headers and refuses anything it cannot account for, on
    the grounds that a preprocessor that silently mishandles a nesting it did
    not expect is exactly the failure this function was written to prevent.
    """
    out = []
    # depth of #if nesting we are inside while discarding
    dead = 0
    nest = 0
    for line in text.splitlines():
        s = line.strip()
        if re.match(r'#\s*if\s+0\s*$', s):
            if dead:
                raise SystemExit('nested #if 0 -- refusing to guess')
            dead = 1
            nest = 0
            continue
        if dead:
            if re.match(r'#\s*if', s):
                nest += 1
            elif re.match(r'#\s*else\s*$', s) and nest == 0:
                dead = 0
            elif re.match(r'#\s*endif', s):
                if nest == 0:
                    dead = 0
                else:
                    nest -= 1
            continue
        out.append(line)
    return '\n'.join(out)


def value_of(tok):
    """Resolve the value forms these headers actually use."""
    tok = tok.strip()
    m = re.fullmatch(r'BIT\((\d+)\)', tok)
    if m:
        return 1 << int(m.group(1))
    m = re.fullmatch(r'0x[0-9a-fA-F]+', tok)
    if m:
        return int(tok, 16)
    m = re.fullmatch(r'\d+', tok)
    if m:
        return int(tok)
    return None


def defines(text, names):
    """Pull `#define NAME value` for each requested name, exactly once."""
    found = {}
    for line in text.splitlines():
        m = re.match(r'#\s*define\s+(\w+)\s+([^/]+?)\s*(?:/\*.*)?$', line)
        if not m:
            continue
        name, raw = m.group(1), m.group(2)
        if name not in names:
            continue
        v = value_of(raw)
        if v is None:
            raise SystemExit('%s = %r is not a form this understands' % (name, raw))
        if name in found and found[name] != v:
            raise SystemExit('%s defined twice with different values after '
                             'preprocessing: 0x%x and 0x%x' % (name, found[name], v))
        found[name] = v
    missing = [n for n in names if n not in found]
    if missing:
        raise SystemExit('not found: %s' % ', '.join(missing))
    return found


def bitfields(text, struct_name):
    """Offsets and widths of a little-endian C bitfield, by declaration order.

    The first declared member occupies the least significant bits, so the
    offset of each is the sum of the widths before it, restarting at every
    32-bit word boundary. Words are separated by blank lines in this struct,
    but that is a formatting accident, so the boundary is taken from the
    running total instead.
    """
    m = re.search(r'struct\s+%s\s*\{(.*?)\n\};' % struct_name, text, re.S)
    if not m:
        raise SystemExit('struct %s not found' % struct_name)
    body = m.group(1)
    # Keep only the __LITTLE_ENDIAN arm.
    if '#ifdef __LITTLE_ENDIAN' in body:
        body = body.split('#ifdef __LITTLE_ENDIAN', 1)[1].split('#else', 1)[0]
    out = {}
    bit = 0
    for fm in re.finditer(r'\bu32\s+(\w+)\s*:\s*(\d+)\s*;', body):
        name, width = fm.group(1), int(fm.group(2))
        word, off = divmod(bit, 32)
        if off + width > 32:
            raise SystemExit('%s straddles a word boundary -- layout not understood' % name)
        out[name] = (word, off, width)
        bit += width
    if not out:
        raise SystemExit('no bitfields parsed from %s' % struct_name)
    return out


REGS = [
    'REG_FPGA0_XA_HSSI_PARM1', 'REG_FPGA0_XA_HSSI_PARM2',
    'REG_FPGA0_XA_LSSI_PARM', 'REG_FPGA0_XA_LSSI_READBACK',
    'REG_HSPI_XA_READBACK',
    'FPGA0_HSSI_PARM1_PI',
    'FPGA0_HSSI_PARM2_ADDR_SHIFT', 'FPGA0_HSSI_PARM2_ADDR_MASK',
    'FPGA0_HSSI_PARM2_EDGE_READ',
    'FPGA0_LSSI_PARM_ADDR_SHIFT', 'FPGA0_LSSI_PARM_DATA_MASK',
]

TXD = [
    'TXDESC_OWN', 'TXDESC_FIRST_SEGMENT', 'TXDESC_LAST_SEGMENT',
    'TXDESC_BROADMULTICAST',
    'TXDESC_QUEUE_SHIFT', 'TXDESC_QUEUE_MASK', 'TXDESC_QUEUE_MGNT',
    'TXDESC_QUEUE_BEACON', 'TXDESC_QUEUE_BE', 'TXDESC_QUEUE_VO',
    'TXDESC32_SEQ_SHIFT', 'TXDESC32_USE_DRIVER_RATE',
    'TXDESC32_RETRY_LIMIT_ENABLE', 'TXDESC32_RETRY_LIMIT_SHIFT',
]

# The RX fields the receive path actually needs to find a frame in a buffer.
RXF = ['pktlen', 'crc32', 'icverr', 'drvinfo_sz', 'shift', 'phy_stats']


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    d = sys.argv[1]
    emit = None
    if '--emit' in sys.argv:
        emit = sys.argv[sys.argv.index('--emit') + 1]

    src = {}
    for f in ('regs.h', 'rtl8xxxu.h'):
        path = os.path.join(d, f)
        raw = open(path, 'rb').read()
        src[f] = strip_dead(raw.decode('utf-8', 'replace'))
        print('%-14s %7d bytes  sha256 %s' % (f, len(raw), hashlib.sha256(raw).hexdigest()[:16]))

    regs = defines(src['regs.h'], REGS)
    txd = defines(src['rtl8xxxu.h'], TXD)
    rxf = bitfields(src['rtl8xxxu.h'], 'rtl8xxxu_rxdesc16')

    print('\n-- RF path A registers and the LSSI/HSSI fields --')
    for k in REGS:
        print('   %-32s 0x%08x' % (k, regs[k]))
    print('\n-- TX descriptor --')
    for k in TXD:
        print('   %-32s 0x%08x' % (k, txd[k]))
    print('\n-- RX descriptor bitfields (word, shift, width) --')
    for k in RXF:
        if k not in rxf:
            raise SystemExit('rx field %s not found' % k)
        print('   %-32s word %d  bit %2d  width %2d' % ((k,) + rxf[k]))

    # A cross-check that costs nothing and would have caught the #if 0 trap:
    # txdw0 is a u8 at byte 3 of the first word, so OWN as a byte bit must be
    # 7 and never 31.
    if txd['TXDESC_OWN'] != 1 << 7:
        raise SystemExit('TXDESC_OWN is 0x%x; txdw0 is a u8 so it must be BIT(7). '
                         'A dead #if 0 branch was probably picked up.' % txd['TXDESC_OWN'])
    # pktlen must be the bottom of word 0, or the receive path reads garbage.
    if rxf['pktlen'] != (0, 0, 14):
        raise SystemExit('rxdesc pktlen is %r, expected word 0 bit 0 width 14' % (rxf['pktlen'],))
    print('\nboth cross-checks pass')

    if emit:
        write_rust(emit, regs, txd, rxf)
        print('appended to %s' % emit)


def write_rust(path, regs, txd, rxf):
    L = []
    a = L.append
    a('')
    a('// --- RF serial interface, path A -------------------------------------')
    a('//')
    a('// Same provenance as the tables above: Linux, rtl8xxxu, regs.h and')
    a('// rtl8xxxu.h, extracted by tools/rtlconv.py rather than retyped.')
    a('//')
    a('// The radio is not memory mapped. Its registers are reached by writing')
    a('// an address and a value together into one baseband register, which is')
    a('// why RADIOA_INIT cannot be applied the way MAC_INIT is.')
    for k in REGS:
        a('pub const %s: u32 = 0x%X;' % (k, regs[k]))
    a('')
    a('// --- TX descriptor ---------------------------------------------------')
    a('//')
    a('// The 8188EU uses the 32-byte descriptor (rtl8xxxu_txdesc32). Word 0 is')
    a('// pkt_size as a little-endian u16, then pkt_offset and txdw0 as single')
    a('// bytes, so the flags below are bits of a *byte* and OWN is bit 7.')
    a('// rtl8xxxu.h defines OWN twice; the BIT(31) form is inside an #if 0 and')
    a('// is not this one.')
    for k in TXD:
        a('pub const %s: u32 = 0x%X;' % (k, txd[k]))
    a('')
    a('// --- RX descriptor ---------------------------------------------------')
    a('//')
    a('// A C bitfield in the original, so these offsets appear nowhere in the')
    a('// source as numbers: under little-endian the first member takes the')
    a('// least significant bits, and the rest follow. Computed, not counted.')
    a('//')
    a('// Each entry is (word, shift, width).')
    for k in RXF:
        w, s, n = rxf[k]
        a('pub const RXDESC_%s: (u32, u32, u32) = (%d, %d, %d);' % (k.upper(), w, s, n))
    a('')
    a('/// Bytes of RX descriptor before the frame. `drvinfo_sz` is counted in')
    a('/// eight-byte units and `shift` is a byte count, both added on top.')
    a('pub const RXDESC16_SIZE: usize = 24;')
    a('')
    with open(path, 'a', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(L))


if __name__ == '__main__':
    main()
