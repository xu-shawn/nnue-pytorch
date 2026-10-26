"""Convert .nnue nets between the legacy input layouts and the HalfKA-consistent ones.

    Full_Threats -> Full_Threats_hm    PP_3Wide -> PP_3Wide_hm    Ram_Ram -> Ram_Ram_hm
    K32H3L^ (legacy K32H3^) -> K32H3^            (other components, e.g. K32Q2^, are copied unchanged)

The legacy threat-style families (Full_Threats, PP_3Wide, Ram_Ram) mirror the board horizontally
when the perspective's king is on files e..h; the _hm families mirror it when the king is on a..d,
as HalfKAv2_hm, K32Q2 and K32H3 do. The two orientations differ by a fixed horizontal mirror of
both squares (square ^ 7), so each family's rows are a fixed permutation of the legacy rows. K32H3
holds the legacy K32H3L's rows block-major (state * 32 + king bucket, see model/modules/features/k32h3.py).

The conversion permutes the quantized feature transformer rows and replaces the feature hash in
the two header words that contain it; everything else is copied byte for byte, so the converted
net evaluates exactly like the source on an engine for the new layout. --reverse converts back.

usage: python scripts/convert_hm_layout.py [--reverse] [--features F] src.nnue dst.nnue
       (F: the source net's feature set, default Full_Threats+PP_3Wide+Ram_Ram+K32H3L^; in the forward direction a
        legacy name K32H3^ is read as K32H3L^,
        or Full_Threats_hm+PP_3Wide_hm+Ram_Ram_hm+K32H3^ with --reverse)
"""

import argparse
import struct
import sys

import numpy as np

# name: (hash, rows, bytes per weight)
COMPONENTS = {
    "Full_Threats": (0x2E6B9D04, 59808, 1),
    "PP_3Wide": (0x86F2B1DD, 4560, 1),
    "Ram_Ram": (0x3A9E1C03, 780, 1),
    "K32H3L^": (0x32B5E2E3, 67584, 1),  # legacy interleaved K32H3 (nets of the ram-k32h3 / heavy branches)
    "K32Q2^": (0x32B5E284, 45056, 1),
    "Full_Threats_hm": (0x2C41EAEA, 59808, 1),
    "PP_3Wide_hm": (0x5C80FA69, 4560, 1),
    "Ram_Ram_hm": (0x5E9FCAC8, 780, 1),
    "K32H3^": (0xAB005CEF, 67584, 1),
}
TO_NEW = {
    "Full_Threats": "Full_Threats_hm",
    "PP_3Wide": "PP_3Wide_hm",
    "Ram_Ram": "Ram_Ram_hm",
    "K32H3L^": "K32H3^",
}
TO_OLD = {v: k for k, v in TO_NEW.items()}


def combine_hash(names):
    h = 0
    for n in names:
        h = ((h << 1) | (h >> 31)) & 0xFFFFFFFF
        h ^= COMPONENTS[n][0]
    return h


# ---- Full_Threats index space (as training_data_loader.cpp / the engine's full_threats.cpp) ----
PAWN, KNIGHT, BISHOP, ROOK, QUEEN = range(5)
NUM_VALID_TARGETS = [4, 10, 8, 8, 10]
# [attacker type][attacked type] -> slot, -1 when the pair is not a feature
SLOT = [
    [-1, 0, -1, 1, -1],
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3, -1],
    [0, 1, 2, 3, -1],
    [0, 1, 2, 3, 4],
]


def _ray_attacks(sq, dirs):
    out = []
    f0, r0 = sq % 8, sq // 8
    for df, dr in dirs:
        f, r = f0 + df, r0 + dr
        while 0 <= f < 8 and 0 <= r < 8:
            out.append(r * 8 + f)
            f, r = f + df, r + dr
    return out


def _step_attacks(sq, steps):
    f0, r0 = sq % 8, sq // 8
    return [
        (r0 + dr) * 8 + f0 + df
        for df, dr in steps
        if 0 <= f0 + df < 8 and 0 <= r0 + dr < 8
    ]


def pseudo_attacks(color, pt, sq):
    """Empty-board attacks, sorted. Pawns attack only from ranks 2..7."""
    if pt == PAWN:
        if not 8 <= sq < 56:
            return []
        dr = 1 if color == 0 else -1
        return sorted(_step_attacks(sq, [(-1, dr), (1, dr)]))
    if pt == KNIGHT:
        return sorted(
            _step_attacks(
                sq,
                [
                    (1, 2),
                    (2, 1),
                    (2, -1),
                    (1, -2),
                    (-1, -2),
                    (-2, -1),
                    (-2, 1),
                    (-1, 2),
                ],
            )
        )
    diag = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    orth = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    dirs = {BISHOP: diag, ROOK: orth, QUEEN: diag + orth}[pt]
    return sorted(_ray_attacks(sq, dirs))


def threat_tables():
    """Per attacker (color * 5 + type): piece offset, per-target block size, per-square offsets."""
    piece_off, block, sq_off = [], [], []
    total = 0
    for color in (0, 1):
        for pt in range(5):
            offs, o = [], 0
            for sq in range(64):
                offs.append(o)
                o += len(pseudo_attacks(color, pt, sq))
            piece_off.append(total)
            block.append(o)
            sq_off.append(offs)
            total += NUM_VALID_TARGETS[pt] * o
    assert total == 59808, total
    return piece_off, block, sq_off


def threat_index(tables, a_color, a_type, frm, to, d_color, d_type):
    """Index of (attacker, from, to, attacked) in oriented coordinates, without deduplication."""
    piece_off, block, sq_off = tables
    a = a_color * 5 + a_type
    slot = SLOT[a_type][d_type]
    assert slot >= 0
    rank = pseudo_attacks(a_color, a_type, frm).index(to)
    return (
        piece_off[a]
        + (d_color * (NUM_VALID_TARGETS[a_type] // 2) + slot) * block[a]
        + sq_off[a][frm]
        + rank
    )


def full_threats_perm():
    """perm[legacy index] = _hm index. Mirrors both squares; a same-type pair is a feature only
    with from > to (the other direction is deduplicated), so a pair on one rank, whose order the
    mirror reverses, also swaps the attacker and the attacked piece. An involution."""
    tables = threat_tables()
    perm = np.full(59808, -1, dtype=np.int64)
    for a_color in (0, 1):
        for a_type in range(5):
            for d_color in (0, 1):
                for d_type in range(5):
                    if SLOT[a_type][d_type] < 0:
                        continue
                    dedup = a_type == d_type and (a_color != d_color or a_type != PAWN)
                    for frm in range(64):
                        for to in pseudo_attacks(a_color, a_type, frm):
                            i = threat_index(
                                tables, a_color, a_type, frm, to, d_color, d_type
                            )
                            f2, t2 = frm ^ 7, to ^ 7
                            if dedup and (frm < to) != (f2 < t2):
                                j = threat_index(
                                    tables, d_color, d_type, t2, f2, a_color, a_type
                                )
                            else:
                                j = threat_index(
                                    tables, a_color, a_type, f2, t2, d_color, d_type
                                )
                            assert perm[i] == -1
                            perm[i] = j
    return perm


def tri(hi, lo):
    return hi * (hi - 1) // 2 + lo


def pair_perm(n_ids, mirror_id):
    """perm[legacy index] = _hm index for a triangular index over unordered pairs of ids."""
    perm = np.full(n_ids * (n_ids - 1) // 2, -1, dtype=np.int64)
    for hi in range(n_ids):
        for lo in range(hi):
            a, b = mirror_id(hi), mirror_id(lo)
            perm[tri(hi, lo)] = tri(max(a, b), min(a, b))
    return perm


def pp_3wide_perm():
    # pawn id = 48 * relative color + oriented square - a2 (squares a2..h7)
    return pair_perm(96, lambda i: (i // 48) * 48 + (((i % 48) + 8) ^ 7) - 8)


def ram_ram_perm():
    # ram id = oriented square of the ram's lower pawn - a2 (squares a2..h6): mirroring is id ^ 7
    return pair_perm(40, lambda i: i ^ 7)


def k32h3_perm():
    """perm[legacy row] = K32H3 row. Legacy buckets: kb * 2 + opp_queen (0..63, some rook or queen
    on the board), 64 + kb (none); K32H3 buckets: state * 32 + kb. 704 rows per bucket."""
    bucket = np.empty(96, dtype=np.int64)
    for kb in range(32):
        bucket[kb * 2 + 0] = (
            1 * 32 + kb
        )  # rook or queen on the board, no opponent queen
        bucket[kb * 2 + 1] = 2 * 32 + kb  # opponent queen
        bucket[64 + kb] = 0 * 32 + kb  # no rook or queen
    return (bucket[:, None] * 704 + np.arange(704)[None, :]).reshape(-1)


PERMS = {
    "Full_Threats": full_threats_perm,
    "PP_3Wide": pp_3wide_perm,
    "Ram_Ram": ram_ram_perm,
    "K32H3L^": k32h3_perm,
}


def get_perm(old_name):
    perm = PERMS[old_name]()
    n = COMPONENTS[old_name][1]
    assert perm.shape == (n,) and np.array_equal(np.sort(perm), np.arange(n)), old_name
    return perm


def convert(buf, src_features, reverse=False, description=None):
    names = src_features.split("+")
    mapping = TO_OLD if reverse else TO_NEW
    for n in names:
        if n not in COMPONENTS:
            raise ValueError(f"unsupported component {n}")
        if n in (TO_NEW if reverse else TO_OLD):
            raise ValueError(f"{n} is already in the target layout")
    dst_names = [mapping.get(n, n) for n in names]
    old_hash, new_hash = combine_hash(names), combine_hash(dst_names)

    version, net_hash, dlen = struct.unpack_from("<III", buf, 0)
    desc = buf[12 : 12 + dlen]
    o = 12 + dlen
    (ft_hash,) = struct.unpack_from("<I", buf, o)
    l1 = (ft_hash ^ old_hash) // 2
    if (ft_hash ^ old_hash) % 2 or not (16 <= l1 <= 16384 and l1 % 16 == 0):
        raise ValueError(
            f"feature transformer hash {ft_hash:#010x} does not match {src_features}"
        )

    if description is not None:
        desc = description.encode()
    out = bytearray(
        struct.pack("<III", version, net_hash ^ old_hash ^ new_hash, len(desc))
    )
    out += desc
    out += struct.pack("<I", ft_hash ^ old_hash ^ new_hash)
    o += 4
    out += buf[o : o + 2 * l1]  # biases (int16)
    o += 2 * l1
    for n in names:
        _, rows, width = COMPONENTS[n]
        size = rows * l1 * width
        seg = np.frombuffer(buf, dtype=np.uint8, count=size, offset=o).reshape(
            rows, l1 * width
        )
        o += size
        old_name = TO_OLD.get(n, n) if reverse else n
        if old_name in PERMS:
            perm = get_perm(old_name)
            if reverse:
                seg = seg[perm]  # new row perm[i] -> old row i
            else:
                dst = np.empty_like(seg)
                dst[perm] = seg  # old row i -> new row perm[i]
                seg = dst
        out += seg.tobytes()
    if o > len(buf):
        raise ValueError("file too short for this feature set")
    out += buf[o:]  # layer stacks, unchanged
    return bytes(out), "+".join(dst_names), l1


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument(
        "--reverse", action="store_true", help="convert a new-layout net back"
    )
    p.add_argument("--features", default=None, help="feature set of the source net")
    p.add_argument("--description", default=None, help="replace the net description")
    a = p.parse_args()
    feats = a.features or (
        "Full_Threats_hm+PP_3Wide_hm+Ram_Ram_hm+K32H3^"
        if a.reverse
        else "Full_Threats+PP_3Wide+Ram_Ram+K32H3L^"
    )
    if not a.reverse:  # legacy nets were exported as K32H3^, which now names the block-major layout
        feats = "+".join("K32H3L^" if x == "K32H3^" else x for x in feats.split("+"))
    with open(a.src, "rb") as f:
        buf = f.read()
    out, dst_feats, l1 = convert(buf, feats, a.reverse, a.description)
    with open(a.dst, "wb") as f:
        f.write(out)
    print(
        f"{a.src} ({feats}, L1 {l1}) -> {a.dst} ({dst_feats}), {len(out)} bytes",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
