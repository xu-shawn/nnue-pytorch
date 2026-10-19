#include "training_data_loader_internal.h"

#include <iostream>
#include <algorithm>
#include <iterator>
#include <future>
#include <random>
#include <cstring>
#include <cassert>
#include <cmath>

#include "lib/rng.h"

using namespace binpack;
using namespace chess;

// ---------------------------------------------------------
// Internal extractors and threat arrays
// ---------------------------------------------------------

static Square orient_flip_2(Color color, Square sq, Square ksq) {
    bool h = ksq.file() < fileE;
    if (color == Color::Black)
        sq = sq.flippedVertically();
    if (h)
        sq = sq.flippedHorizontally();
    return sq;
}

struct HalfKAv2_hm {
    static constexpr std::string_view NAME = "HalfKAv2_hm";

    static constexpr int NUM_SQ              = 64;
    static constexpr int NUM_PT              = 12;
    static constexpr int NUM_PLANES          = NUM_SQ * NUM_PT;
    static constexpr int INPUTS              = NUM_PLANES * NUM_SQ / 2;
    static constexpr int MAX_ACTIVE_FEATURES = 32;

    // clang-format off
    static constexpr int KingBuckets[64] = {
      -1, -1, -1, -1, 31, 30, 29, 28,
      -1, -1, -1, -1, 27, 26, 25, 24,
      -1, -1, -1, -1, 23, 22, 21, 20,
      -1, -1, -1, -1, 19, 18, 17, 16,
      -1, -1, -1, -1, 15, 14, 13, 12,
      -1, -1, -1, -1, 11, 10, 9, 8,
      -1, -1, -1, -1, 7, 6, 5, 4,
      -1, -1, -1, -1, 3, 2, 1, 0
    };
    // clang-format on

    static int feature_index(Color color, Square ksq, Square sq, Piece p) {
        Square o_ksq = orient_flip_2(color, ksq, ksq);
        auto   p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
        return static_cast<int>(orient_flip_2(color, sq, ksq)) + p_idx * NUM_SQ
             + KingBuckets[static_cast<int>(o_ksq)] * NUM_PLANES;
    }

    static std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) {
        auto& pos    = e.pos;
        auto  pieces = pos.piecesBB();
        auto  ksq    = pos.kingSquare(color);

        int j = 0;
        for (Square sq : pieces)
        {
            auto p      = pos.pieceAt(sq);
            features[j] = feature_index(color, ksq, sq, p);
            ++j;
        }
        return {j, INPUTS};
    }
};

struct HalfKAv2_hmExtractor: IFeatureExtractor {
    int inputs() const override { return HalfKAv2_hm::INPUTS; }
    int max_active_features() const override { return HalfKAv2_hm::MAX_ACTIVE_FEATURES; }
    std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) const override {
        return HalfKAv2_hm::fill_features_sparse(e, features, color);
    }
};

struct K32Q2 {
    static constexpr std::string_view NAME = "K32Q2";

    static constexpr int NUM_SQ              = 64;
    static constexpr int NUM_PT              = 12;
    static constexpr int NUM_PLANES          = NUM_SQ * NUM_PT;
    static constexpr int NUM_KING_BUCKETS    = NUM_SQ / 2;
    static constexpr int NUM_QUEEN_BUCKETS   = 2;
    static constexpr int NUM_BUCKETS         = NUM_KING_BUCKETS * NUM_QUEEN_BUCKETS;
    static constexpr int INPUTS              = NUM_PLANES * NUM_BUCKETS;
    static constexpr int MAX_ACTIVE_FEATURES = 32;

    // clang-format off
    static constexpr int KingBuckets[64] = {
      -1, -1, -1, -1, 31, 30, 29, 28,
      -1, -1, -1, -1, 27, 26, 25, 24,
      -1, -1, -1, -1, 23, 22, 21, 20,
      -1, -1, -1, -1, 19, 18, 17, 16,
      -1, -1, -1, -1, 15, 14, 13, 12,
      -1, -1, -1, -1, 11, 10, 9, 8,
      -1, -1, -1, -1, 7, 6, 5, 4,
      -1, -1, -1, -1, 3, 2, 1, 0
    };
    // clang-format on

    static int feature_index(Color color, Square ksq, bool opp_queen, Square sq, Piece p) {
        Square o_ksq  = orient_flip_2(color, ksq, ksq);
        auto   p_idx  = static_cast<int>(p.type()) * 2 + (p.color() != color);
        int    bucket = KingBuckets[static_cast<int>(o_ksq)] * 2 + (opp_queen ? 1 : 0);
        return static_cast<int>(orient_flip_2(color, sq, ksq)) + p_idx * NUM_SQ
             + bucket * NUM_PLANES;
    }

    static std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) {
        auto& pos       = e.pos;
        auto  pieces    = pos.piecesBB();
        auto  ksq       = pos.kingSquare(color);
        bool  opp_queen = pos.piecesBB(Piece(PieceType::Queen, !color)).any();

        int j = 0;
        for (Square sq : pieces)
        {
            auto p      = pos.pieceAt(sq);
            features[j] = feature_index(color, ksq, opp_queen, sq, p);
            ++j;
        }
        return {j, INPUTS};
    }
};

struct K32Q2Extractor: IFeatureExtractor {
    int inputs() const override { return K32Q2::INPUTS; }
    int max_active_features() const override { return K32Q2::MAX_ACTIVE_FEATURES; }
    std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) const override {
        return K32Q2::fill_features_sparse(e, features, color);
    }
};

// Pawn attackers no longer target pawns; those relationships are represented by PP_3Wide.
constexpr int numvalidtargets[6] = {4, 10, 8, 8, 10, 0};

constexpr int threat_piece(Piece piece) { return (int) piece.color() * 5 + (int) piece.type(); }

constexpr std::uint64_t pawn_pair_mask(int sq) {
    constexpr std::uint64_t FileA = 0x0101010101010101ULL;
    constexpr std::uint64_t Rank1 = 0xFFULL;
    constexpr std::uint64_t Rank8 = Rank1 << 56;

    const int     f     = sq & 7;
    std::uint64_t files = FileA << f;
    if (f > 0)
        files |= FileA << (f - 1);
    if (f < 7)
        files |= FileA << (f + 1);
    return files & ~Rank1 & ~Rank8 & ~(1ULL << sq);
}

using ThreatOffsetTable = std::array<std::array<int, 66>, 10>;

struct ThreatFeatureCalculation {
    ThreatOffsetTable table;
    int               totalfeatures;
};

constexpr auto threatfeaturecalc = []() {
    ThreatOffsetTable t{};

    constexpr auto pseudo_attacks = bb::detail::generatePseudoAttacks();
    int            pieceoffset    = 0;

    Piece piecetbl[10] = {whitePawn, whiteKnight, whiteBishop, whiteRook, whiteQueen,
                          blackPawn, blackKnight, blackBishop, blackRook, blackQueen};

    for (int piece = 0; piece < 10; ++piece)
    {
        t[piece][65]     = pieceoffset;
        int squareoffset = 0;

        for (int from = (int) a1; from <= (int) h8; ++from)
        {
            t[piece][from] = squareoffset;
            if (piecetbl[piece].type() == PieceType::Pawn)
            {
                if (from >= (int) a2 && from <= (int) h7)
                    squareoffset +=
                      bb::pawnAttacks(Bitboard::square(Square(from)), piecetbl[piece].color())
                        .count();
            }
            else
            {
                squareoffset += pseudo_attacks[piecetbl[piece].type()][Square(from)].count();
            }
        }

        t[piece][64] = squareoffset;
        pieceoffset += numvalidtargets[(int) piecetbl[piece].type()] * squareoffset;
    }
    return ThreatFeatureCalculation{t, pieceoffset};
}();

constexpr ThreatOffsetTable threatoffsets  = threatfeaturecalc.table;
constexpr int               threatfeatures = threatfeaturecalc.totalfeatures;
static_assert(threatfeatures == 59808);

struct FullThreats {
    static constexpr std::string_view NAME = "Full_Threats";

    static constexpr int SQUARE_NB           = 64;
    static constexpr int PIECE_NB            = 12;
    static constexpr int COLOR_NB            = 2;
    static constexpr int PIECE_TYPE_NB       = 6;
    static constexpr int MAX_ACTIVE_FEATURES = 128;

    static constexpr int INPUTS = threatfeatures;

    // clang-format off
    static constexpr Square OrientTBL[COLOR_NB][SQUARE_NB] = {
      { a1, a1, a1, a1, h1, h1, h1, h1,
        a1, a1, a1, a1, h1, h1, h1, h1,
        a1, a1, a1, a1, h1, h1, h1, h1,
        a1, a1, a1, a1, h1, h1, h1, h1,
        a1, a1, a1, a1, h1, h1, h1, h1,
        a1, a1, a1, a1, h1, h1, h1, h1,
        a1, a1, a1, a1, h1, h1, h1, h1,
        a1, a1, a1, a1, h1, h1, h1, h1 },
      { a8, a8, a8, a8, h8, h8, h8, h8,
        a8, a8, a8, a8, h8, h8, h8, h8,
        a8, a8, a8, a8, h8, h8, h8, h8,
        a8, a8, a8, a8, h8, h8, h8, h8,
        a8, a8, a8, a8, h8, h8, h8, h8,
        a8, a8, a8, a8, h8, h8, h8, h8,
        a8, a8, a8, a8, h8, h8, h8, h8,
        a8, a8, a8, a8, h8, h8, h8, h8 }
    };

    static constexpr int map[PIECE_TYPE_NB][PIECE_TYPE_NB] = {
      {-1, 0, -1, 1, -1, -1},
      { 0, 1,  2, 3,  4, -1},
      { 0, 1,  2, 3, -1, -1},
      { 0, 1,  2, 3, -1, -1},
      { 0, 1,  2, 3,  4, -1},
      {-1,-1, -1,-1, -1, -1}
    };

    // clang-format on

    static int
    threat_index(Color perspective, Piece attkr, Square from, Square to, Piece attkd, Square ksq) {
        from = (Square) ((int) from ^ (int) OrientTBL[(int) perspective][(int) ksq]);
        to   = (Square) ((int) to ^ (int) OrientTBL[(int) perspective][(int) ksq]);

        if (perspective == Color::Black)
        {
            attkr = Piece::fromId((int) attkr ^ 1);
            attkd = Piece::fromId((int) attkd ^ 1);
        }

        int slot = map[(int) attkr.type()][(int) attkd.type()];
        if (slot < 0)
            return -1;

        bool enemy = attkr.color() != attkd.color();
        if (attkr.type() == attkd.type() && (enemy || attkr.type() != PieceType::Pawn) && from < to)
            return -1;

        Bitboard attacks  = attkr.type() == PieceType::Pawn
                            ? bb::pawnAttacks(Bitboard::square(from), attkr.color())
                            : bb::detail::pseudoAttacks()[attkr.type()][from];
        int      attacker = threat_piece(attkr);
        return int(threatoffsets[attacker][65]
                   + (int(attkd.color()) * (numvalidtargets[(int) attkr.type()] / 2) + slot)
                       * threatoffsets[attacker][64]
                   + threatoffsets[attacker][(int) from]
                   + (Bitboard::fromBits((1ULL << (int) to) - 1) & attacks).count());
    }

    static std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) {
        auto& pos    = e.pos;
        auto  pieces = pos.piecesBB();
        // Kings are never targets; exclude them upfront
        auto  piecesNoK   = pieces & ~pos.piecesBB(whiteKing) & ~pos.piecesBB(blackKing);
        auto  ksq         = pos.kingSquare(color);
        Color order[2][2] = {{Color::White, Color::Black}, {Color::Black, Color::White}};
        int   k           = 0;

        for (int i = (int) Color::White; i <= (int) Color::Black; i++)
        {
            Color c = order[(int) color][i];

            // ---- Pawn attacks ----
            {
                Piece    attkr    = Piece(PieceType::Pawn, c);
                Bitboard bb       = pos.piecesBB(attkr);
                auto     allPawns = pos.piecesBB(whitePawn) | pos.piecesBB(blackPawn);

                // Diagonal pawn threats exclude pawn targets (those are handled by PP_3Wide)
                auto diagTargets = piecesNoK & ~allPawns;

                auto right   = (c == Color::White) ? Offset(1, 1) : Offset(-1, -1);
                auto left    = (c == Color::White) ? Offset(-1, 1) : Offset(1, -1);
                int  r_delta = (c == Color::White) ? 9 : -9;
                int  l_delta = (c == Color::White) ? 7 : -7;

                for (Square to : bb.shifted(right) & diagTargets)
                {
                    Square from  = Square((int) to - r_delta);
                    Piece  attkd = pos.pieceAt(to);
                    int    index = threat_index(color, attkr, from, to, attkd, ksq);
                    if (index >= 0)
                    {
                        features[k++] = index;
                    }
                }
                for (Square to : bb.shifted(left) & diagTargets)
                {
                    Square from  = Square((int) to - l_delta);
                    Piece  attkd = pos.pieceAt(to);
                    int    index = threat_index(color, attkr, from, to, attkd, ksq);
                    if (index >= 0)
                    {
                        features[k++] = index;
                    }
                }
            }

            for (int j = (int) PieceType::Knight; j < (int) PieceType::King; j++)
            {
                Piece    attkr = Piece(PieceType(j), c);
                Bitboard bb    = pos.piecesBB(attkr);

                for (Square from : bb)
                {
                    Bitboard attacks = pos.attacks(from) & piecesNoK;
                    for (Square to : attacks)
                    {
                        Piece attkd = pos.pieceAt(to);
                        int   index = threat_index(color, attkr, from, to, attkd, ksq);
                        if (index >= 0)
                        {
                            features[k++] = index;
                        }
                    }
                }
            }
        }
        return {k, INPUTS};
    }
};

struct FullThreatsExtractor: IFeatureExtractor {
    int inputs() const override { return FullThreats::INPUTS; }
    int max_active_features() const override { return FullThreats::MAX_ACTIVE_FEATURES; }
    std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) const override {
        return FullThreats::fill_features_sparse(e, features, color);
    }
};

// MAX_ACTIVE_FEATURES only reserves batch slots; real positions have at most about 40
// pawn pairs. 112 leaves room for Pawn_Push or Passed_48 (16 slots each), so that
// Full_Threats + PP_3Wide + one of them + K32Q2 reserve 288 slots, the width the GPU
// feature-transformer kernels are tuned for.
struct PP_3Wide {
    static constexpr std::string_view NAME = "PP_3Wide";

    static constexpr int PAWN_IDS            = 2 * 48;
    static constexpr int INPUTS              = PAWN_IDS * (PAWN_IDS - 1) / 2;
    static constexpr int MAX_ACTIVE_FEATURES = 112;

    static int make_pawn_id(Color color, Square square) {
        return 48 * static_cast<int>(color) + static_cast<int>(square) - static_cast<int>(a2);
    }

    static int make_index(
      Color perspective, Color color, Square from, Square to, Color paired_color, Square ksq) {
        int orient = static_cast<int>(
          FullThreats::OrientTBL[static_cast<int>(perspective)][static_cast<int>(ksq)]);
        Square from_oriented = static_cast<Square>(static_cast<int>(from) ^ orient);
        Square to_oriented   = static_cast<Square>(static_cast<int>(to) ^ orient);

        Color color_oriented =
          static_cast<Color>(static_cast<int>(color) ^ static_cast<int>(perspective));
        Color paired_color_oriented =
          static_cast<Color>(static_cast<int>(paired_color) ^ static_cast<int>(perspective));

        if (from_oriented < a2 || from_oriented > h7 || to_oriented < a2 || to_oriented > h7)
            return INPUTS;

        int id_a = make_pawn_id(color_oriented, from_oriented);
        int id_b = make_pawn_id(paired_color_oriented, to_oriented);
        int hi   = std::max(id_a, id_b);
        int lo   = std::min(id_a, id_b);

        return hi * (hi - 1) / 2 + lo;
    }

    static std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) {
        auto& pos         = e.pos;
        auto  allPawns    = pos.piecesBB(whitePawn) | pos.piecesBB(blackPawn);
        auto  ksq         = pos.kingSquare(color);
        Color order[2][2] = {{Color::White, Color::Black}, {Color::Black, Color::White}};
        int   k           = 0;

        for (int i = static_cast<int>(Color::White); i <= static_cast<int>(Color::Black); ++i)
        {
            Color    c  = order[static_cast<int>(color)][i];
            Piece    p  = Piece(PieceType::Pawn, c);
            Bitboard bb = pos.piecesBB(p);

            for (Square from : bb)
            {
                Bitboard targets =
                  Bitboard::fromBits(pawn_pair_mask(static_cast<int>(from))) & allPawns;
                for (Square to : targets)
                {
                    if (from < to)
                        continue;

                    Color paired_color = pos.pieceAt(to).color();
                    int   index        = make_index(color, c, from, to, paired_color, ksq);
                    if (index < INPUTS)
                    {
                        features[k] = index;
                        ++k;
                    }
                }
            }
        }

        return {k, INPUTS};
    }
};

struct PP_3WideExtractor: IFeatureExtractor {
    int inputs() const override { return PP_3Wide::INPUTS; }
    int max_active_features() const override { return PP_3Wide::MAX_ACTIVE_FEATURES; }
    std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) const override {
        return PP_3Wide::fill_features_sparse(e, features, color);
    }
};

// Pawn pairs two files apart (complementing PP_3Wide, which pairs pawns on the same or an
// adjacent file), with PP_3Wide's pair index over all 96 pawn ids. At most 64 pairs are
// active: the most is 8 pawns on each of two files two apart.
struct PP_Wide2 {
    static constexpr std::string_view NAME = "PP_Wide2";

    static constexpr int INPUTS              = PP_3Wide::INPUTS;
    static constexpr int MAX_ACTIVE_FEATURES = 64;

    static constexpr std::uint64_t pair_mask(int sq) {
        constexpr std::uint64_t FileA = 0x0101010101010101ULL;
        constexpr std::uint64_t Rank1 = 0xFFULL;
        constexpr std::uint64_t Rank8 = Rank1 << 56;

        const int     f     = sq & 7;
        std::uint64_t files = 0;
        if (f > 1)
            files |= FileA << (f - 2);
        if (f < 6)
            files |= FileA << (f + 2);
        return files & ~Rank1 & ~Rank8;
    }

    static std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) {
        auto& pos      = e.pos;
        auto  allPawns = pos.piecesBB(whitePawn) | pos.piecesBB(blackPawn);
        auto  ksq      = pos.kingSquare(color);
        int   k        = 0;

        for (Color c : {Color::White, Color::Black})
            for (Square from : pos.piecesBB(Piece(PieceType::Pawn, c)))
                for (Square to : Bitboard::fromBits(pair_mask(static_cast<int>(from))) & allPawns)
                {
                    if (from < to)
                        continue;
                    int index =
                      PP_3Wide::make_index(color, c, from, to, pos.pieceAt(to).color(), ksq);
                    if (index < INPUTS)
                        features[k++] = index;
                }

        return {k, INPUTS};
    }
};

struct PP_Wide2Extractor: IFeatureExtractor {
    int inputs() const override { return PP_Wide2::INPUTS; }
    int max_active_features() const override { return PP_Wide2::MAX_ACTIVE_FEATURES; }
    std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) const override {
        return PP_Wide2::fill_features_sparse(e, features, color);
    }
};

// The features below share the orientation of Full_Threats and PP_3Wide: squares are
// xor-ed with FullThreats::OrientTBL[perspective][ksq] and colors are relative to the
// perspective.

// A pawn whose push square holds a non-pawn piece of either color (pawns blocked by
// pawns are covered by PP_3Wide). Index: pawn color, blocker piece (5 types per color),
// pawn square.
struct PawnPush {
    static constexpr std::string_view NAME = "Pawn_Push";

    static constexpr int INPUTS              = 2 * 10 * 48;
    static constexpr int MAX_ACTIVE_FEATURES = 16;

    static int
    make_index(Color perspective, Color pawn_color, Square from, Piece blocker, int orient) {
        int kind = (static_cast<int>(blocker.color()) ^ static_cast<int>(perspective)) * 5
                 + static_cast<int>(blocker.type()) - static_cast<int>(PieceType::Knight);
        int color_oriented = static_cast<int>(pawn_color) ^ static_cast<int>(perspective);
        return (color_oriented * 10 + kind) * 48 + (static_cast<int>(from) ^ orient) - 8;
    }

    static std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) {
        auto& pos = e.pos;
        auto  ksq = pos.kingSquare(color);
        int   orient =
          static_cast<int>(FullThreats::OrientTBL[static_cast<int>(color)][static_cast<int>(ksq)]);
        auto allPawns = pos.piecesBB(whitePawn) | pos.piecesBB(blackPawn);
        auto blockers = pos.piecesBB() & ~allPawns;
        int  k        = 0;

        for (Color c : {Color::White, Color::Black})
        {
            Bitboard pawns = pos.piecesBB(Piece(PieceType::Pawn, c));
            int      push  = c == Color::White ? 8 : -8;
            for (Square to : pawns.shiftedVertically(c == Color::White ? 1 : -1) & blockers)
                features[k++] = make_index(color, c, Square(static_cast<int>(to) - push),
                                           pos.pieceAt(to), orient);
        }

        return {k, INPUTS};
    }
};

struct PawnPushExtractor: IFeatureExtractor {
    int inputs() const override { return PawnPush::INPUTS; }
    int max_active_features() const override { return PawnPush::MAX_ACTIVE_FEATURES; }
    std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) const override {
        return PawnPush::fill_features_sparse(e, features, color);
    }
};

// Attackers of the promotion square of every pawn on its 7th rank, counted per attacker
// kind relative to the pawn's color: own pawn, knight, bishop, rook, queen, king, then
// enemy knight, bishop, rook, queen, king (enemy pawns cannot attack it). Each kind has
// two thermometer levels, one or more attackers and two or more.
// Index: ((pawn color * 8 + promotion file) * 11 + kind) * 2 + level.
// A pawn has at most 20 features (11 own and 9 enemy levels) and at most 16 pawns
// stand on their 7th rank, hence 320 slots; real positions use about 6.
struct PromoAtk {
    static constexpr std::string_view NAME = "Promo_Atk";

    static constexpr int KINDS               = 11;
    static constexpr int INPUTS              = 2 * 8 * KINDS * 2;
    static constexpr int MAX_ACTIVE_FEATURES = 16 * 20;

    static std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) {
        auto& pos = e.pos;
        auto  ksq = pos.kingSquare(color);
        int   flip =
          static_cast<int>(FullThreats::OrientTBL[static_cast<int>(color)][static_cast<int>(ksq)])
          & 7;
        int k = 0;

        for (Color c : {Color::White, Color::Black})
        {
            Bitboard pawns   = pos.piecesBB(Piece(PieceType::Pawn, c));
            Bitboard seventh = Bitboard::rank(c == Color::White ? rank7 : rank2);

            for (Square from : pawns& seventh)
            {
                Square   to    = Square(static_cast<int>(from) + (c == Color::White ? 8 : -8));
                Bitboard own   = pos.attackers(to, c);
                Bitboard enemy = pos.attackers(to, !c);
                int      base  = ((static_cast<int>(c) ^ static_cast<int>(color)) * 8
                            + (static_cast<int>(to.file()) ^ flip))
                         * KINDS;

                int counts[KINDS];
                for (int t = 0; t < 6; ++t)
                    counts[t] = (own & pos.piecesBB(Piece(PieceType(t), c))).count();
                for (int t = 1; t < 6; ++t)
                    counts[5 + t] = (enemy & pos.piecesBB(Piece(PieceType(t), !c))).count();

                for (int kind = 0; kind < KINDS; ++kind)
                    for (int level = 0; level < std::min(counts[kind], 2); ++level)
                        features[k++] = (base + kind) * 2 + level;
            }
        }

        return {k, INPUTS};
    }
};

struct PromoAtkExtractor: IFeatureExtractor {
    int inputs() const override { return PromoAtk::INPUTS; }
    int max_active_features() const override { return PromoAtk::MAX_ACTIVE_FEATURES; }
    std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) const override {
        return PromoAtk::fill_features_sparse(e, features, color);
    }
};

// Like Promo_Atk, but every attacker of a promotion square is its own feature, indexed by
// pawn color, promotion file, attacker kind (as in Promo_Atk) and attacker square. Only
// squares from which that kind can attack the promotion square get an index (1620 total).
struct PromoAtkSq {
    static constexpr std::string_view NAME = "Promo_AtkSq";

    static constexpr int KINDS = 11;
    // A promotion square has at most 17 attackers (2 pawns, 8 knights, 5 sliders on the
    // rays that stay on the board, 2 kings) and at most 16 pawns stand on their 7th rank.
    static constexpr int MAX_ACTIVE_FEATURES = 16 * 17;

    struct Table {
        std::uint64_t reach[2][8][KINDS];
        int           offset[2][8][KINDS];
        int           total;
    };

    // Indexed by pawn color relative to the perspective (0: the perspective's pawns, which
    // promote on the oriented 8th rank), oriented promotion file and kind.
    static constexpr Table table = []() {
        Table          t{};
        constexpr auto pseudo = bb::detail::generatePseudoAttacks();
        int            offset = 0;
        for (int c = 0; c < 2; ++c)
            for (int f = 0; f < 8; ++f)
                for (int k = 0; k < KINDS; ++k)
                {
                    Square        to = Square(File(f), c == 0 ? rank8 : rank1);
                    std::uint64_t reach;
                    if (k == 0)
                        reach = bb::pawnAttacks(Bitboard::square(to),
                                                c == 0 ? Color::Black : Color::White)
                                  .bits();
                    else
                        reach = pseudo[PieceType(k <= 5 ? k : k - 5)][to].bits();
                    t.reach[c][f][k]  = reach;
                    t.offset[c][f][k] = offset;
                    offset += Bitboard::fromBits(reach).count();
                }
        t.total = offset;
        return t;
    }();

    static constexpr int INPUTS = table.total;
    static_assert(INPUTS == 1620);

    static std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) {
        auto& pos = e.pos;
        auto  ksq = pos.kingSquare(color);
        int   orient =
          static_cast<int>(FullThreats::OrientTBL[static_cast<int>(color)][static_cast<int>(ksq)]);
        int k = 0;

        for (Color c : {Color::White, Color::Black})
        {
            Bitboard pawns   = pos.piecesBB(Piece(PieceType::Pawn, c));
            Bitboard seventh = Bitboard::rank(c == Color::White ? rank7 : rank2);
            int      rel     = static_cast<int>(c) ^ static_cast<int>(color);

            for (Square from : pawns& seventh)
            {
                Square to   = Square(static_cast<int>(from) + (c == Color::White ? 8 : -8));
                int    file = (static_cast<int>(to) ^ orient) & 7;

                for (Color side : {c, !c})
                    for (Square a : pos.attackers(to, side))
                    {
                        int type = static_cast<int>(pos.pieceAt(a).type());
                        int kind = side == c ? type : type + 5;
                        int a_o  = static_cast<int>(a) ^ orient;
                        assert(table.reach[rel][file][kind] >> a_o & 1);
                        features[k++] =
                          table.offset[rel][file][kind]
                          + Bitboard::fromBits(table.reach[rel][file][kind] & ((1ULL << a_o) - 1))
                              .count();
                    }
            }
        }

        return {k, INPUTS};
    }
};

struct PromoAtkSqExtractor: IFeatureExtractor {
    int inputs() const override { return PromoAtkSq::INPUTS; }
    int max_active_features() const override { return PromoAtkSq::MAX_ACTIVE_FEATURES; }
    std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) const override {
        return PromoAtkSq::fill_features_sparse(e, features, color);
    }
};

// Attackers of the empty square in front of every pawn (its stop square), like Promo_Atk
// but for all pawns. Attacker kinds relative to the pawn: own knight, bishop, rook, queen,
// king, then the same five for the enemy; two thermometer levels (>= 1, >= 2). Pawn
// attackers are left out because PP_3Wide already has every pawn pair on adjacent files,
// and occupied stop squares because the threat features already cover attacks on the
// blocker. Index: ((pawn colour * 48 + pawn square - 8) * 10 + kind) * 2 + level, colour
// relative to the perspective and the pawn square oriented like the threat features.
struct StopAtk {
    static constexpr std::string_view NAME = "Stop_Atk";

    static constexpr int KINDS               = 10;
    static constexpr int INPUTS              = 2 * 48 * KINDS * 2;
    static constexpr int MAX_ACTIVE_FEATURES = 16 * KINDS * 2;

    static std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) {
        auto& pos    = e.pos;
        auto  ksq    = pos.kingSquare(color);
        int   orient = static_cast<int>(
          FullThreats::OrientTBL[static_cast<int>(color)][static_cast<int>(ksq)]);
        Bitboard occupied = pos.piecesBB();
        int      k        = 0;

        for (Color c : {Color::White, Color::Black})
        {
            for (Square from : pos.piecesBB(Piece(PieceType::Pawn, c)))
            {
                Square to = Square(static_cast<int>(from) + (c == Color::White ? 8 : -8));
                if (occupied.isSet(to))
                    continue;

                Bitboard own   = pos.attackers(to, c);
                Bitboard enemy = pos.attackers(to, !c);
                int      base  = ((static_cast<int>(c) ^ static_cast<int>(color)) * 48
                            + ((static_cast<int>(from) ^ orient) - 8))
                         * KINDS;

                int counts[KINDS];
                for (int t = 0; t < 5; ++t)
                {
                    counts[t]     = (own & pos.piecesBB(Piece(PieceType(t + 1), c))).count();
                    counts[5 + t] = (enemy & pos.piecesBB(Piece(PieceType(t + 1), !c))).count();
                }

                for (int kind = 0; kind < KINDS; ++kind)
                    for (int level = 0; level < std::min(counts[kind], 2); ++level)
                        features[k++] = (base + kind) * 2 + level;
            }
        }

        return {k, INPUTS};
    }
};

struct StopAtkExtractor: IFeatureExtractor {
    int inputs() const override { return StopAtk::INPUTS; }
    int max_active_features() const override { return StopAtk::MAX_ACTIVE_FEATURES; }
    std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) const override {
        return StopAtk::fill_features_sparse(e, features, color);
    }
};

// Threat features for the empty square in front of every pawn (its stop square): every
// knight, bishop, rook, queen or king attacking it is a feature of its own, indexed by
// attacker square, which extends the threat features (occupied squares only, no king
// attackers) to these squares. Pawn attackers and occupied stop squares are left out as in
// Stop_Atk. Index: pawn colour relative to the perspective, oriented pawn square (a2 to
// h7), attacker kind (own knight, bishop, rook, queen, king, then the enemy's, relative to
// the pawn) and attacker square, counting only the squares from which that kind can reach
// the stop square.
struct StopThreats {
    static constexpr std::string_view NAME = "Stop_Threats";

    static constexpr int KINDS = 10;
    // A square has at most 18 non-pawn attackers (8 knights, a slider on each of its 8
    // rays and 2 kings) and there are at most 16 pawns; real positions use about 10.
    static constexpr int MAX_ACTIVE_FEATURES = 16 * 18;

    struct Table {
        std::uint64_t reach[2][48][KINDS];
        int           offset[2][48][KINDS];
        int           total;
    };

    // Indexed by pawn colour relative to the perspective (0: the perspective's pawns, which
    // advance towards the oriented 8th rank), oriented pawn square - 8 and kind.
    static constexpr Table table = []() {
        Table          t{};
        constexpr auto pseudo = bb::detail::generatePseudoAttacks();
        int            offset = 0;
        for (int c = 0; c < 2; ++c)
            for (int s = 0; s < 48; ++s)
                for (int k = 0; k < KINDS; ++k)
                {
                    Square        to    = Square(s + 8 + (c == 0 ? 8 : -8));
                    std::uint64_t reach = pseudo[PieceType(k % 5 + 1)][to].bits();
                    t.reach[c][s][k]    = reach;
                    t.offset[c][s][k]   = offset;
                    offset += Bitboard::fromBits(reach).count();
                }
        t.total = offset;
        return t;
    }();

    static constexpr int INPUTS = table.total;
    static_assert(INPUTS == 11256);

    static std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) {
        auto& pos = e.pos;
        auto  ksq = pos.kingSquare(color);
        int   orient =
          static_cast<int>(FullThreats::OrientTBL[static_cast<int>(color)][static_cast<int>(ksq)]);
        Bitboard occupied = pos.piecesBB();
        Bitboard pawns    = pos.piecesBB(whitePawn) | pos.piecesBB(blackPawn);
        int      k        = 0;

        for (Color c : {Color::White, Color::Black})
        {
            int rel = static_cast<int>(c) ^ static_cast<int>(color);

            for (Square from : pos.piecesBB(Piece(PieceType::Pawn, c)))
            {
                Square to = Square(static_cast<int>(from) + (c == Color::White ? 8 : -8));
                if (occupied.isSet(to))
                    continue;

                int s = (static_cast<int>(from) ^ orient) - 8;
                for (Color side : {c, !c})
                    for (Square a : pos.attackers(to, side) & ~pawns)
                    {
                        int kind = static_cast<int>(pos.pieceAt(a).type()) - 1 + (side == c ? 0 : 5);
                        int a_o  = static_cast<int>(a) ^ orient;
                        assert(table.reach[rel][s][kind] >> a_o & 1);
                        features[k++] =
                          table.offset[rel][s][kind]
                          + Bitboard::fromBits(table.reach[rel][s][kind] & ((1ULL << a_o) - 1))
                              .count();
                    }
            }
        }

        return {k, INPUTS};
    }
};

struct StopThreatsExtractor: IFeatureExtractor {
    int inputs() const override { return StopThreats::INPUTS; }
    int max_active_features() const override { return StopThreats::MAX_ACTIVE_FEATURES; }
    std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) const override {
        return StopThreats::fill_features_sparse(e, features, color);
    }
};

// Threat features for the empty squares attacked by pawns: for each color, every empty
// square its pawns attack is a target, and every knight, bishop, rook, queen or king
// attacking it is a feature of its own, indexed by attacker square (kinds as in
// Stop_Threats, relative to the pawns' color). Two pawn kinds add what PP_3Wide misses:
// the target is attacked by two of the color's pawns (pawns two files apart, e.g. c4 and
// e4 on d5), and each enemy pawn attacking it, by square. Index: pawn color relative to
// the perspective, oriented target square, kind and attacker square, counting only the
// squares from which that kind can attack the target.
struct PawnAtkThreats {
    static constexpr std::string_view NAME = "PawnAtk_Threats";

    static constexpr int KINDS = 12;
    // At most 32 targets (16 per color), each with at most 18 non-pawn attackers, the
    // double-attack feature and 2 enemy pawns; real positions use about 15.
    static constexpr int MAX_ACTIVE_FEATURES = 32 * 21;

    struct Table {
        std::uint64_t reach[2][64][KINDS];
        int           offset[2][64][KINDS];
        int           total;
    };

    // Indexed by pawn color relative to the perspective (0: the perspective's pawns, which
    // advance towards the oriented 8th rank), oriented target square and kind.
    static constexpr Table table = []() {
        Table                   t{};
        constexpr auto          pseudo     = bb::detail::generatePseudoAttacks();
        constexpr std::uint64_t PawnRanks  = 0x00FFFFFFFFFFFF00ULL;
        int                     offset     = 0;
        for (int c = 0; c < 2; ++c)
            for (int s = 0; s < 64; ++s)
            {
                // Pawns on ranks 2 to 7 attack ranks 3 to 8 (color 0) or 1 to 6 (color 1)
                int  rank   = s / 8;
                bool target = c == 0 ? rank >= 2 : rank <= 5;
                for (int k = 0; k < KINDS; ++k)
                {
                    std::uint64_t reach = 0;
                    if (target && k < 10)
                        reach = pseudo[PieceType(k % 5 + 1)][Square(s)].bits();
                    else if (target && k == 10)
                        reach = 1ULL << s;
                    else if (target && k == 11)
                        reach = bb::pawnAttacks(Bitboard::square(Square(s)),
                                                c == 0 ? Color::White : Color::Black)
                                  .bits()
                              & PawnRanks;
                    t.reach[c][s][k]  = reach;
                    t.offset[c][s][k] = offset;
                    offset += Bitboard::fromBits(reach).count();
                }
            }
        t.total = offset;
        return t;
    }();

    static constexpr int INPUTS = table.total;
    static_assert(INPUTS == 11464);

    static int index(int rel, int target, int kind, int a_o) {
        assert(table.reach[rel][target][kind] >> a_o & 1);
        return table.offset[rel][target][kind]
             + Bitboard::fromBits(table.reach[rel][target][kind] & ((1ULL << a_o) - 1)).count();
    }

    static std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) {
        auto& pos = e.pos;
        auto  ksq = pos.kingSquare(color);
        int   orient =
          static_cast<int>(FullThreats::OrientTBL[static_cast<int>(color)][static_cast<int>(ksq)]);
        Bitboard occupied = pos.piecesBB();
        Bitboard pawns    = pos.piecesBB(whitePawn) | pos.piecesBB(blackPawn);
        int      k        = 0;

        for (Color c : {Color::White, Color::Black})
        {
            int      rel     = static_cast<int>(c) ^ static_cast<int>(color);
            Bitboard own     = pos.piecesBB(Piece(PieceType::Pawn, c));
            Bitboard enemy   = pos.piecesBB(Piece(PieceType::Pawn, !c));
            Bitboard targets = bb::pawnAttacks(own, c) & ~occupied;

            for (Square t : targets)
            {
                int t_o = static_cast<int>(t) ^ orient;
                for (Color side : {c, !c})
                    for (Square a : pos.attackers(t, side) & ~pawns)
                    {
                        int kind = static_cast<int>(pos.pieceAt(a).type()) - 1 + (side == c ? 0 : 5);
                        features[k++] = index(rel, t_o, kind, static_cast<int>(a) ^ orient);
                    }
                if ((bb::pawnAttacks(Bitboard::square(t), !c) & own).count() >= 2)
                    features[k++] = index(rel, t_o, 10, t_o);
                for (Square a : bb::pawnAttacks(Bitboard::square(t), c) & enemy)
                    features[k++] = index(rel, t_o, 11, static_cast<int>(a) ^ orient);
            }
        }

        return {k, INPUTS};
    }
};

struct PawnAtkThreatsExtractor: IFeatureExtractor {
    int inputs() const override { return PawnAtkThreats::INPUTS; }
    int max_active_features() const override { return PawnAtkThreats::MAX_ACTIVE_FEATURES; }
    std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) const override {
        return PawnAtkThreats::fill_features_sparse(e, features, color);
    }
};

// Stop_Threats restricted to passed pawns (no enemy pawn ahead on the same or an adjacent
// file), with Stop_Threats' index. Passers are rare, so these features change far less often.
struct PassStopThreats {
    static constexpr std::string_view NAME = "PassStop_Threats";

    static constexpr int INPUTS              = StopThreats::INPUTS;
    static constexpr int MAX_ACTIVE_FEATURES = StopThreats::MAX_ACTIVE_FEATURES;

    static std::uint64_t passed(Color c, std::uint64_t white, std::uint64_t black) {
        constexpr std::uint64_t FileA = 0x0101010101010101ULL;
        constexpr std::uint64_t FileH = FileA << 7;

        std::uint64_t enemy = c == Color::White ? black : white;
        std::uint64_t span  = enemy | ((enemy << 1) & ~FileA) | ((enemy >> 1) & ~FileH);
        std::uint64_t fill;
        if (c == Color::White)
        {
            fill = span >> 8;
            fill |= fill >> 8;
            fill |= fill >> 16;
            fill |= fill >> 32;
        }
        else
        {
            fill = span << 8;
            fill |= fill << 8;
            fill |= fill << 16;
            fill |= fill << 32;
        }
        return (c == Color::White ? white : black) & ~fill;
    }

    static std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) {
        auto& pos = e.pos;
        auto  ksq = pos.kingSquare(color);
        int   orient =
          static_cast<int>(FullThreats::OrientTBL[static_cast<int>(color)][static_cast<int>(ksq)]);
        Bitboard occupied = pos.piecesBB();
        Bitboard pawns    = pos.piecesBB(whitePawn) | pos.piecesBB(blackPawn);
        const std::uint64_t white = pos.piecesBB(whitePawn).bits(), black = pos.piecesBB(blackPawn).bits();
        int k = 0;

        for (Color c : {Color::White, Color::Black})
        {
            int rel = static_cast<int>(c) ^ static_cast<int>(color);

            for (Square from : Bitboard::fromBits(passed(c, white, black)))
            {
                Square to = Square(static_cast<int>(from) + (c == Color::White ? 8 : -8));
                if (occupied.isSet(to))
                    continue;

                int s = (static_cast<int>(from) ^ orient) - 8;
                for (Color side : {c, !c})
                    for (Square a : pos.attackers(to, side) & ~pawns)
                    {
                        int kind = static_cast<int>(pos.pieceAt(a).type()) - 1 + (side == c ? 0 : 5);
                        int a_o  = static_cast<int>(a) ^ orient;
                        features[k++] =
                          StopThreats::table.offset[rel][s][kind]
                          + Bitboard::fromBits(StopThreats::table.reach[rel][s][kind] & ((1ULL << a_o) - 1))
                              .count();
                    }
            }
        }

        return {k, INPUTS};
    }
};

struct PassStopThreatsExtractor: IFeatureExtractor {
    int inputs() const override { return PassStopThreats::INPUTS; }
    int max_active_features() const override { return PassStopThreats::MAX_ACTIVE_FEATURES; }
    std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) const override {
        return PassStopThreats::fill_features_sparse(e, features, color);
    }
};

// PassStop_Threats only while at most N knights, bishops, rooks and queens of both colors are
// left. In an engine the features then cost nothing in most of the game.
template<int N>
struct PassStopThreatsGated {
    static constexpr int INPUTS              = PassStopThreats::INPUTS;
    static constexpr int MAX_ACTIVE_FEATURES = PassStopThreats::MAX_ACTIVE_FEATURES;

    static std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) {
        auto& pos    = e.pos;
        int   pieces = pos.piecesBB().count()
                   - (pos.piecesBB(whitePawn) | pos.piecesBB(blackPawn)).count() - 2;
        if (pieces > N)
            return {0, INPUTS};
        return PassStopThreats::fill_features_sparse(e, features, color);
    }
};

template<int N>
struct PassStopThreatsGatedExtractor: IFeatureExtractor {
    int inputs() const override { return PassStopThreatsGated<N>::INPUTS; }
    int max_active_features() const override { return PassStopThreatsGated<N>::MAX_ACTIVE_FEATURES; }
    std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) const override {
        return PassStopThreatsGated<N>::fill_features_sparse(e, features, color);
    }
};

// Rams: a white pawn with a black pawn directly in front of it. From each perspective a ram is
// identified by the oriented square of the lower of its two pawns (the perspective's own pawn),
// on ranks 2 to 6: 40 ram ids. Orientation as in Full_Threats.
struct Rams {
    int           count = 0;
    int           id[8];
    std::uint64_t members[8];

    Rams(const Position& pos, int orient) {
        const std::uint64_t white = pos.piecesBB(whitePawn).bits(), black = pos.piecesBB(blackPawn).bits();
        for (Square w : Bitboard::fromBits(white & (black >> 8)))
        {
            int a = static_cast<int>(w) ^ orient, b = (static_cast<int>(w) + 8) ^ orient;
            id[count]      = std::min(a, b) - 8;
            members[count] = (1ULL << static_cast<int>(w)) | (1ULL << (static_cast<int>(w) + 8));
            ++count;
        }
    }
};

static int ram_orient(const Position& pos, Color color) {
    return static_cast<int>(
      FullThreats::OrientTBL[static_cast<int>(color)][static_cast<int>(pos.kingSquare(color))]);
}

// Every ram paired with every other pawn on the board. Index: ram id * 96 + other pawn (color
// relative to the perspective * 48 + oriented square - 8). At most 8 rams with 14 other pawns.
struct RamPawn {
    static constexpr std::string_view NAME = "Ram_Pawn";

    static constexpr int INPUTS              = 40 * 96;
    static constexpr int MAX_ACTIVE_FEATURES = 8 * 14;

    static std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) {
        auto&      pos    = e.pos;
        const int  orient = ram_orient(pos, color);
        const Rams rams(pos, orient);
        int        k = 0;

        for (int i = 0; i < rams.count; ++i)
            for (Color c : {Color::White, Color::Black})
            {
                int rel = static_cast<int>(c) ^ static_cast<int>(color);
                for (Square p : Bitboard::fromBits(pos.piecesBB(Piece(PieceType::Pawn, c)).bits()
                                                   & ~rams.members[i]))
                    features[k++] = rams.id[i] * 96 + rel * 48 + (static_cast<int>(p) ^ orient) - 8;
            }

        return {k, INPUTS};
    }
};

// Every ram paired with the files that hold another pawn of each color. Index: ram id * 16 +
// color relative to the perspective * 8 + oriented file. Pawn pushes leave these unchanged.
struct RamPawnFile {
    static constexpr std::string_view NAME = "Ram_PawnFile";

    static constexpr int INPUTS              = 40 * 16;
    static constexpr int MAX_ACTIVE_FEATURES = 8 * 16;

    static std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) {
        auto&      pos    = e.pos;
        const int  orient = ram_orient(pos, color);
        const Rams rams(pos, orient);
        int        k = 0;

        for (int i = 0; i < rams.count; ++i)
        {
            unsigned files = 0;
            for (Color c : {Color::White, Color::Black})
            {
                int rel = static_cast<int>(c) ^ static_cast<int>(color);
                for (Square p : Bitboard::fromBits(pos.piecesBB(Piece(PieceType::Pawn, c)).bits()
                                                   & ~rams.members[i]))
                    files |= 1u << (rel * 8 + ((static_cast<int>(p) ^ orient) & 7));
            }
            for (int b = 0; b < 16; ++b)
                if (files & (1u << b))
                    features[k++] = rams.id[i] * 16 + b;
        }

        return {k, INPUTS};
    }
};

// Every pair of rams on the board, by both ram ids (triangular index). These change only when
// a ram forms or breaks.
struct RamRam {
    static constexpr std::string_view NAME = "Ram_Ram";

    static constexpr int INPUTS              = 40 * 39 / 2;
    static constexpr int MAX_ACTIVE_FEATURES = 8 * 7 / 2;

    static std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) {
        auto&      pos = e.pos;
        const Rams rams(pos, ram_orient(pos, color));
        int        k = 0;

        for (int i = 0; i < rams.count; ++i)
            for (int j = 0; j < rams.count; ++j)
                if (rams.id[i] < rams.id[j])
                    features[k++] = rams.id[j] * (rams.id[j] - 1) / 2 + rams.id[i];

        return {k, INPUTS};
    }
};

template<typename T>
struct SimpleExtractor: IFeatureExtractor {
    int inputs() const override { return T::INPUTS; }
    int max_active_features() const override { return T::MAX_ACTIVE_FEATURES; }
    std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) const override {
        return T::fill_features_sparse(e, features, color);
    }
};

// Pawn-piece pairs: every knight, bishop, rook, queen or king on the same or an adjacent file
// as a pawn, at any rank. Index: pawn (colour relative to the perspective, oriented square),
// piece kind (colour relative to the perspective, type) and the piece's square within the
// pawn's three-file window. At most 16 pieces with 16 pawns, hence 256 slots.
struct PawnPiece3Wide {
    static constexpr std::string_view NAME = "PawnPiece_3Wide";

    static constexpr int KINDS               = 10;
    static constexpr int MAX_ACTIVE_FEATURES = 256;

    struct Table {
        int offset[2][64];        // first feature of each pawn (colour, oriented square)
        int width[64];            // window squares per pawn square
        std::uint64_t window[64]; // squares of the window (pawn square excluded)
        int total;
    };

    static constexpr Table table = []() {
        Table t{};
        constexpr std::uint64_t FileA = 0x0101010101010101ULL;
        int offset = 0;
        for (int sq = 0; sq < 64; ++sq)
        {
            const int     f     = sq & 7;
            std::uint64_t files = FileA << f;
            if (f > 0)
                files |= FileA << (f - 1);
            if (f < 7)
                files |= FileA << (f + 1);
            t.window[sq] = files & ~(1ULL << sq);
            t.width[sq]  = Bitboard::fromBits(t.window[sq]).count();
        }
        for (int c = 0; c < 2; ++c)
            for (int sq = 8; sq < 56; ++sq)
            {
                t.offset[c][sq] = offset;
                offset += KINDS * t.width[sq];
            }
        t.total = offset;
        return t;
    }();

    static constexpr int INPUTS = table.total;
    static_assert(INPUTS == 20160);

    static std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) {
        auto& pos = e.pos;
        auto  ksq = pos.kingSquare(color);
        int   orient =
          static_cast<int>(FullThreats::OrientTBL[static_cast<int>(color)][static_cast<int>(ksq)]);
        Bitboard pieces = pos.piecesBB() & ~(pos.piecesBB(whitePawn) | pos.piecesBB(blackPawn));
        int      k      = 0;

        for (Color c : {Color::White, Color::Black})
        {
            int rel = static_cast<int>(c) ^ static_cast<int>(color);
            for (Square p : pos.piecesBB(Piece(PieceType::Pawn, c)))
            {
                int p_o = static_cast<int>(p) ^ orient;
                for (Square q : pieces)
                {
                    int q_o = static_cast<int>(q) ^ orient;
                    if (!(table.window[p_o] >> q_o & 1))
                        continue;
                    Piece pc   = pos.pieceAt(q);
                    int   kind = (static_cast<int>(pc.color()) ^ static_cast<int>(color)) * 5
                             + static_cast<int>(pc.type()) - 1;
                    features[k++] =
                      table.offset[rel][p_o] + kind * table.width[p_o]
                      + Bitboard::fromBits(table.window[p_o] & ((1ULL << q_o) - 1)).count();
                }
            }
        }

        return {k, INPUTS};
    }
};

struct PawnPiece3WideExtractor: IFeatureExtractor {
    int inputs() const override { return PawnPiece3Wide::INPUTS; }
    int max_active_features() const override { return PawnPiece3Wide::MAX_ACTIVE_FEATURES; }
    std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) const override {
        return PawnPiece3Wide::fill_features_sparse(e, features, color);
    }
};

// Threats on each king's zone: every knight, bishop, rook or queen (of either colour)
// attacking the king or an empty square next to it (the threat features cover attacks on
// occupied squares except kings). Index: king colour relative to the perspective, oriented
// target square, attacker kind relative to the king (own N, B, R, Q, then the enemy's) and
// attacker square, counting only the squares from which that kind reaches the target.
struct KingZoneThreats {
    static constexpr std::string_view NAME = "KingZone_Threats";

    static constexpr int KINDS = 8;
    // 2 kings x 9 targets x at most 16 attackers (8 knights and a slider on each ray)
    static constexpr int MAX_ACTIVE_FEATURES = 288;

    struct Table {
        std::uint64_t reach[64][KINDS];
        int           offset[2][64][KINDS];
        int           total;
    };

    static constexpr Table table = []() {
        Table          t{};
        constexpr auto pseudo = bb::detail::generatePseudoAttacks();
        int            offset = 0;
        for (int c = 0; c < 2; ++c)
            for (int s = 0; s < 64; ++s)
                for (int k = 0; k < KINDS; ++k)
                {
                    std::uint64_t reach = pseudo[PieceType(k % 4 + 1)][Square(s)].bits();
                    t.reach[s][k]       = reach;
                    t.offset[c][s][k]   = offset;
                    offset += Bitboard::fromBits(reach).count();
                }
        t.total = offset;
        return t;
    }();

    static constexpr int INPUTS = table.total;
    static_assert(INPUTS == 12992);

    static std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) {
        constexpr auto pseudo = bb::detail::generatePseudoAttacks();
        auto&          pos    = e.pos;
        auto           ksq    = pos.kingSquare(color);
        int            orient =
          static_cast<int>(FullThreats::OrientTBL[static_cast<int>(color)][static_cast<int>(ksq)]);
        Bitboard occupied = pos.piecesBB();
        Bitboard minors   = occupied & ~(pos.piecesBB(whitePawn) | pos.piecesBB(blackPawn)
                                       | pos.piecesBB(whiteKing) | pos.piecesBB(blackKing));
        int      k        = 0;

        for (Color c : {Color::White, Color::Black})
        {
            int      rel     = static_cast<int>(c) ^ static_cast<int>(color);
            Square   king    = pos.kingSquare(c);
            Bitboard targets = (pseudo[PieceType::King][king] & ~occupied) | Bitboard::square(king);
            for (Square t : targets)
            {
                int t_o = static_cast<int>(t) ^ orient;
                for (Square a : (pos.attackers(t, Color::White) | pos.attackers(t, Color::Black)) & minors)
                {
                    Piece pc   = pos.pieceAt(a);
                    int   kind = static_cast<int>(pc.type()) - 1 + (pc.color() == c ? 0 : 4);
                    int   a_o  = static_cast<int>(a) ^ orient;
                    features[k++] =
                      table.offset[rel][t_o][kind]
                      + Bitboard::fromBits(table.reach[t_o][kind] & ((1ULL << a_o) - 1)).count();
                }
            }
        }

        return {k, INPUTS};
    }
};

struct KingZoneThreatsExtractor: IFeatureExtractor {
    int inputs() const override { return KingZoneThreats::INPUTS; }
    int max_active_features() const override { return KingZoneThreats::MAX_ACTIVE_FEATURES; }
    std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) const override {
        return KingZoneThreats::fill_features_sparse(e, features, color);
    }
};

// Subsets of PawnPiece_3Wide and PawnAtk_Threats with the same indices: pawn-king and
// pawn-rook pairs only, and the enemy-pawn contacts (kind 11) of PawnAtk_Threats only.
struct PawnKR3Wide {
    static constexpr std::string_view NAME = "PawnKR_3Wide";

    static constexpr int INPUTS              = PawnPiece3Wide::INPUTS;
    static constexpr int MAX_ACTIVE_FEATURES = PawnPiece3Wide::MAX_ACTIVE_FEATURES;

    static std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) {
        int tmp[PawnPiece3Wide::MAX_ACTIVE_FEATURES];
        auto [n, inputs] = PawnPiece3Wide::fill_features_sparse(e, tmp, color);
        int k = 0;
        for (int i = 0; i < n; ++i)
        {
            // kind = (piece colour ^ perspective) * 5 + type - 1: rook 2, king 4
            const auto& t    = PawnPiece3Wide::table;
            int         base = 0;
            // recover the kind from the index: find the pawn block containing it
            int idx = tmp[i];
            for (int c = 0; c < 2 && !base; ++c)
                for (int sq = 8; sq < 56; ++sq)
                {
                    int lo = t.offset[c][sq], hi = lo + PawnPiece3Wide::KINDS * t.width[sq];
                    if (idx >= lo && idx < hi)
                    {
                        int kind = (idx - lo) / t.width[sq] % 5;
                        if (kind == 2 || kind == 4)
                            features[k++] = idx;
                        base = 1;
                        break;
                    }
                }
        }
        return {k, inputs};
    }
};

struct PawnKR3WideExtractor: IFeatureExtractor {
    int inputs() const override { return PawnKR3Wide::INPUTS; }
    int max_active_features() const override { return PawnKR3Wide::MAX_ACTIVE_FEATURES; }
    std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) const override {
        return PawnKR3Wide::fill_features_sparse(e, features, color);
    }
};

struct PawnContacts {
    static constexpr std::string_view NAME = "Pawn_Contacts";

    static constexpr int INPUTS              = PawnAtkThreats::INPUTS;
    static constexpr int MAX_ACTIVE_FEATURES = 64;

    static std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) {
        auto& pos = e.pos;
        auto  ksq = pos.kingSquare(color);
        int   orient =
          static_cast<int>(FullThreats::OrientTBL[static_cast<int>(color)][static_cast<int>(ksq)]);
        Bitboard occupied = pos.piecesBB();
        int      k        = 0;
        for (Color c : {Color::White, Color::Black})
        {
            int      rel   = static_cast<int>(c) ^ static_cast<int>(color);
            Bitboard own   = pos.piecesBB(Piece(PieceType::Pawn, c));
            Bitboard enemy = pos.piecesBB(Piece(PieceType::Pawn, !c));
            for (Square t : bb::pawnAttacks(own, c) & ~occupied)
                for (Square a : bb::pawnAttacks(Bitboard::square(t), c) & enemy)
                    features[k++] = PawnAtkThreats::index(rel, static_cast<int>(t) ^ orient, 11,
                                                          static_cast<int>(a) ^ orient);
        }
        return {k, INPUTS};
    }
};

struct PawnContactsExtractor: IFeatureExtractor {
    int inputs() const override { return PawnContacts::INPUTS; }
    int max_active_features() const override { return PawnContacts::MAX_ACTIVE_FEATURES; }
    std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) const override {
        return PawnContacts::fill_features_sparse(e, features, color);
    }
};

// The perspective's own passed pawns: no enemy pawn ahead on the same or an adjacent file.
// Index: pawn square (ranks 2 to 7).
struct Passed48 {
    static constexpr std::string_view NAME = "Passed_48";

    static constexpr int INPUTS              = 48;
    static constexpr int MAX_ACTIVE_FEATURES = 16;

    static std::uint64_t passed_pawns(Color c, std::uint64_t white, std::uint64_t black) {
        constexpr std::uint64_t FileA = 0x0101010101010101ULL;
        constexpr std::uint64_t FileH = FileA << 7;

        std::uint64_t enemy = c == Color::White ? black : white;
        std::uint64_t span  = enemy | ((enemy << 1) & ~FileA) | ((enemy >> 1) & ~FileH);
        std::uint64_t fill;
        if (c == Color::White)
        {
            fill = span >> 8;
            fill |= fill >> 8;
            fill |= fill >> 16;
            fill |= fill >> 32;
        }
        else
        {
            fill = span << 8;
            fill |= fill << 8;
            fill |= fill << 16;
            fill |= fill << 32;
        }
        return (c == Color::White ? white : black) & ~fill;
    }

    static std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) {
        auto& pos = e.pos;
        auto  ksq = pos.kingSquare(color);
        int   orient =
          static_cast<int>(FullThreats::OrientTBL[static_cast<int>(color)][static_cast<int>(ksq)]);
        auto passed = Bitboard::fromBits(
          passed_pawns(color, pos.piecesBB(whitePawn).bits(), pos.piecesBB(blackPawn).bits()));
        int k = 0;

        for (Square sq : passed)
            features[k++] = (static_cast<int>(sq) ^ orient) - 8;

        return {k, INPUTS};
    }
};

struct Passed48Extractor: IFeatureExtractor {
    int inputs() const override { return Passed48::INPUTS; }
    int max_active_features() const override { return Passed48::MAX_ACTIVE_FEATURES; }
    std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) const override {
        return Passed48::fill_features_sparse(e, features, color);
    }
};

struct ComposedFeatureExtractor: IFeatureExtractor {
    std::vector<std::unique_ptr<IFeatureExtractor>> extractors;
    int                                             m_inputs;
    int                                             m_max_active;

    ComposedFeatureExtractor(std::vector<std::unique_ptr<IFeatureExtractor>> exts) :
        extractors(std::move(exts)),
        m_inputs(0),
        m_max_active(0) {
        for (auto& e : extractors)
        {
            m_inputs += e->inputs();
            m_max_active += e->max_active_features();
        }
    }

    int inputs() const override { return m_inputs; }
    int max_active_features() const override { return m_max_active; }

    std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) const override {
        int total_written = 0;
        int input_offset  = 0;

        for (auto& ext : extractors)
        {
            auto [written, ext_inputs] =
              ext->fill_features_sparse(e, features + total_written, color);

            // Offset the feature indices for this component
            for (int i = 0; i < written; ++i)
                features[total_written + i] += input_offset;

            input_offset += ext_inputs;
            total_written += written;
        }

        return {total_written, m_inputs};
    }
};

static std::unique_ptr<IFeatureExtractor> make_single_extractor(std::string_view name) {
    if (name == "HalfKAv2_hm")
        return std::make_unique<HalfKAv2_hmExtractor>();
    if (name == "K32Q2")
        return std::make_unique<K32Q2Extractor>();
    if (name == "Full_Threats")
        return std::make_unique<FullThreatsExtractor>();
    if (name == "PP_3Wide")
        return std::make_unique<PP_3WideExtractor>();
    if (name == "Pawn_Push")
        return std::make_unique<PawnPushExtractor>();
    if (name == "Promo_Atk")
        return std::make_unique<PromoAtkExtractor>();
    if (name == "Promo_AtkSq")
        return std::make_unique<PromoAtkSqExtractor>();
    if (name == "Stop_Atk")
        return std::make_unique<StopAtkExtractor>();
    if (name == "Stop_Threats")
        return std::make_unique<StopThreatsExtractor>();
    if (name == "PawnAtk_Threats")
        return std::make_unique<PawnAtkThreatsExtractor>();
    if (name == "PP_Wide2")
        return std::make_unique<PP_Wide2Extractor>();
    if (name == "PassStop_Threats")
        return std::make_unique<PassStopThreatsExtractor>();
    if (name == "PassStop_Threats_N4")
        return std::make_unique<PassStopThreatsGatedExtractor<4>>();
    if (name == "PassStop_Threats_N8")
        return std::make_unique<PassStopThreatsGatedExtractor<8>>();
    if (name == "Ram_Pawn")
        return std::make_unique<SimpleExtractor<RamPawn>>();
    if (name == "Ram_PawnFile")
        return std::make_unique<SimpleExtractor<RamPawnFile>>();
    if (name == "Ram_Ram")
        return std::make_unique<SimpleExtractor<RamRam>>();
    if (name == "PawnPiece_3Wide")
        return std::make_unique<PawnPiece3WideExtractor>();
    if (name == "KingZone_Threats")
        return std::make_unique<KingZoneThreatsExtractor>();
    if (name == "PawnKR_3Wide")
        return std::make_unique<PawnKR3WideExtractor>();
    if (name == "Pawn_Contacts")
        return std::make_unique<PawnContactsExtractor>();
    if (name == "Passed_48")
        return std::make_unique<Passed48Extractor>();
    return nullptr;
}

std::shared_ptr<IFeatureExtractor> get_feature(std::string_view name) {
    std::vector<std::unique_ptr<IFeatureExtractor>> components;
    std::size_t                                     start = 0;

    while (start < name.size())
    {
        auto pos  = name.find('+', start);
        auto part = name.substr(start, pos == std::string_view::npos ? pos : pos - start);
        auto ext  = make_single_extractor(part);

        if (!ext)
        {
            std::cerr << "Unknown feature component: " << part << std::endl;
            return nullptr;
        }

        components.push_back(std::move(ext));
        start = (pos == std::string_view::npos) ? name.size() : pos + 1;
    }

    if (components.empty())
        return nullptr;

    if (components.size() == 1)
        return std::shared_ptr<IFeatureExtractor>(std::move(components[0]));

    return std::make_shared<ComposedFeatureExtractor>(std::move(components));
}

// ---------------------------------------------------------
// Class Implementations
// ---------------------------------------------------------

template<typename T>
struct BumpAllocator {
    T* ptr;
    BumpAllocator(T* block) :
        ptr(block) { }
    T* alloc(size_t count) {
        T* res = ptr;
        ptr += count;
        return res;
    }
};

SparseBatchBufferPool::SparseBatchBufferPool(std::size_t floats,
                                             std::size_t ints,
                                             std::size_t capacity) :
    m_floats(floats),
    m_ints(ints),
    m_capacity(capacity) {
    m_free.reserve(capacity);
}

SparseBatchBufferPool::Buffers SparseBatchBufferPool::acquire(std::size_t floats,
                                                              std::size_t ints) {
    if (floats == m_floats && ints == m_ints)
    {
        std::lock_guard lock(m_mutex);
        if (!m_free.empty())
        {
            auto buffers = std::move(m_free.back());
            m_free.pop_back();
            return buffers;
        }
    }
    return {std::unique_ptr<float[]>(new float[floats]), std::unique_ptr<int[]>(new int[ints])};
}

void SparseBatchBufferPool::release(Buffers buffers, std::size_t floats, std::size_t ints) {
    if (floats == m_floats && ints == m_ints)
    {
        std::lock_guard lock(m_mutex);
        if (m_free.size() < m_capacity)
            m_free.push_back(std::move(buffers));
    }
    // Partial final batches and excess buffers are freed normally. The cache
    // never retains storage with a different layout or beyond its capacity.
}

SparseBatch::SparseBatch(const IFeatureExtractor&               feature_set,
                         const std::vector<TrainingDataEntry>&  entries,
                         std::shared_ptr<SparseBatchBufferPool> buffer_pool) :
#ifdef NNUE_LOADER_STATISTICS
    entries_copy(entries),
#endif
    m_buffer_pool(std::move(buffer_pool)) {
    num_inputs                = feature_set.inputs();
    size                      = entries.size();
    max_active_features       = feature_set.max_active_features();
    const size_t total_floats = size * 3;
    const size_t total_ints   = size * 2 + size * max_active_features * 2;

    if (m_buffer_pool)
    {
        auto buffers  = m_buffer_pool->acquire(total_floats, total_ints);
        m_float_block = buffers.first.release();
        m_int_block   = buffers.second.release();
    }
    else
    {
        m_float_block = new float[total_floats];
        m_int_block   = new int[total_ints];
    }

    BumpAllocator<float> float_alloc(m_float_block);
    is_white = float_alloc.alloc(size);
    outcome  = float_alloc.alloc(size);
    score    = float_alloc.alloc(size);

    BumpAllocator<int> int_alloc(m_int_block);
    white        = int_alloc.alloc(size * max_active_features);
    black        = int_alloc.alloc(size * max_active_features);
    piece_count  = int_alloc.alloc(size);
    queen_bucket = int_alloc.alloc(size);

    num_active_white_features = 0;
    num_active_black_features = 0;
    m_feature_counts.resize(std::size_t(size) * 2);

    for (int i = 0; i < size * max_active_features; ++i)
        white[i] = -1;
    for (int i = 0; i < size * max_active_features; ++i)
        black[i] = -1;

    for (int i = 0; i < size; ++i)
        fill_entry(feature_set, i, entries[i]);
}

SparseBatch::~SparseBatch() {
    if (m_buffer_pool)
        m_buffer_pool->release(
          {std::unique_ptr<float[]>(m_float_block), std::unique_ptr<int[]>(m_int_block)},
          std::size_t(size) * 3, std::size_t(size) * (2 * max_active_features + 2));
    else
    {
        delete[] m_float_block;
        delete[] m_int_block;
    }
}

void SparseBatch::fill_entry(const IFeatureExtractor& fs, int i, const TrainingDataEntry& e) {
    is_white[i]     = static_cast<float>(e.pos.sideToMove() == Color::White);
    outcome[i]      = (e.result + 1.0f) / 2.0f;
    score[i]        = e.score;
    piece_count[i]  = e.pos.piecesBB().count();
    const auto stm  = e.pos.sideToMove();
    queen_bucket[i] = (e.pos.piecesBB(Piece(PieceType::Queen, stm)).any() ? 2 : 0)
                    | (e.pos.piecesBB(Piece(PieceType::Queen, !stm)).any() ? 1 : 0);
    fill_features(fs, i, e);
}

void SparseBatch::fill_features(const IFeatureExtractor& fs, int i, const TrainingDataEntry& e) {
    const int offset           = i * max_active_features;
    const int nw               = fs.fill_features_sparse(e, white + offset, Color::White).first;
    const int nb               = fs.fill_features_sparse(e, black + offset, Color::Black).first;
    m_feature_counts[i]        = nw;
    m_feature_counts[size + i] = nb;
    num_active_white_features += nw;
    num_active_black_features += nb;
}

bool SparseBatch::pack_indices(int* output, std::size_t capacity) const {
    const std::size_t required =
      std::size_t(size) * 4 + 2 + num_active_white_features + num_active_black_features;
    if (!output || capacity < required)
        return false;
    int* white_offsets = output;
    int* black_offsets = white_offsets + size + 1;
    int* white_ids     = black_offsets + size + 1;
    int* black_ids     = white_ids + num_active_white_features;
    int* piece_counts  = black_ids + num_active_black_features;
    int  white_count   = 0, black_count = 0;
    white_offsets[0] = black_offsets[0] = 0;
    for (int row = 0; row < size; ++row)
    {
        const int         num_white = m_feature_counts[row];
        const int         num_black = m_feature_counts[size + row];
        const std::size_t offset    = std::size_t(row) * max_active_features;
        std::memcpy(white_ids + white_count, white + offset, std::size_t(num_white) * sizeof(int));
        std::memcpy(black_ids + black_count, black + offset, std::size_t(num_black) * sizeof(int));
        white_count += num_white;
        black_count += num_black;
        white_offsets[row + 1] = white_count;
        black_offsets[row + 1] = black_count;
    }
    std::memcpy(piece_counts, piece_count, std::size_t(size) * sizeof(int));
    std::memcpy(piece_counts + size, queen_bucket, std::size_t(size) * sizeof(int));
    return true;
}

int FeaturedBatchStream::calculate_num_reader_threads(int concurrency) {
    if (worker_thread_ratio >= 1)
        return 1;
    return std::max(1, concurrency - calculate_num_worker_threads(concurrency));
}

int FeaturedBatchStream::calculate_num_worker_threads(int concurrency) {
    if (worker_thread_ratio <= 0)
        return 1;
    return std::max(1, static_cast<int>(std::floor(concurrency * worker_thread_ratio)));
}

FeaturedBatchStream::FeaturedBatchStream(
  std::shared_ptr<IFeatureExtractor>            feature_set,
  int                                           concurrency,
  const std::vector<std::string>&               filenames,
  int                                           batch_size,
  bool                                          cyclic,
  std::function<bool(const TrainingDataEntry&)> skipPredicate,
  int                                           rank,
  int                                           world_size,
  nnue::UniquePositionCounter*                  counter,
  DataloaderIOConfig                            io_config) :
    BaseType(calculate_num_reader_threads(concurrency),
             filenames,
             cyclic,
             skipPredicate,
             counter,
             rank,
             world_size,
             io_config,
             calculate_num_worker_threads(concurrency)),
    m_feature_set(std::move(feature_set)),
    m_batch_size(batch_size),
    m_concurrency(concurrency),
    m_num_workers(calculate_num_worker_threads(concurrency)),
    m_batch_queue_capacity(calculate_num_worker_threads(concurrency) + 4) {

    // At most one fetched batch, the finished deque, and one batch per builder
    // normally exist at a time. Recycle that working set without growing a
    // process-global cache or keeping buffers alive after the stream closes.
    m_buffer_pool = std::make_shared<SparseBatchBufferPool>(
      std::size_t(batch_size) * 3,
      std::size_t(batch_size) * (2 * m_feature_set->max_active_features() + 2),
      std::size_t(m_batch_queue_capacity) + m_num_workers.load() + 1);
    m_stop_flag.store(false);

    auto worker = [this]() {
        std::vector<TrainingDataEntry> entries;
        entries.reserve(m_batch_size);

        while (!m_stop_flag.load())
        {
            entries.clear();
            {
                BaseType::m_stream->fill_threadsafe(entries, m_batch_size);
                if (entries.empty())
                    break;
            }

            auto batch = new SparseBatch(*m_feature_set, entries, m_buffer_pool);

            {
                std::unique_lock lock(m_batch_mutex);
                m_batches_not_full.wait(lock, [this]() {
                    return m_batches.size() < static_cast<size_t>(m_batch_queue_capacity)
                        || m_stop_flag.load();
                });
                m_batches.emplace_back(batch);
                lock.unlock();
                m_batches_any.notify_one();
            }
        }
        m_num_workers.fetch_sub(1);
        m_batches_any.notify_one();
    };

    const int num_worker_threads = calculate_num_worker_threads(concurrency);
    for (int i = 0; i < num_worker_threads; ++i)
    {
        m_workers.emplace_back(worker);
    }
}

FeaturedBatchStream::~FeaturedBatchStream() {
    m_stop_flag.store(true);
    m_batches_not_full.notify_all();
    for (auto& worker : m_workers)
    {
        if (worker.joinable())
            worker.join();
    }
    for (auto& batch : m_batches)
        delete batch;
    // Counter is owned by AnyStream and deleted after m_stream (and its
    // workers) are destroyed, preventing use-after-free.
}

SparseBatch* FeaturedBatchStream::next() {
    std::unique_lock lock(m_batch_mutex);
    m_batches_any.wait(lock, [this]() { return !m_batches.empty() || m_num_workers.load() == 0; });
    if (!m_batches.empty())
    {
        auto batch = m_batches.front();
        m_batches.pop_front();
        lock.unlock();
        m_batches_not_full.notify_one();
        return batch;
    }
    return nullptr;
}

Fen::Fen() :
    m_fen(nullptr) { }

Fen::Fen(const std::string& fen) :
    m_size(fen.size()),
    m_fen(new char[fen.size() + 1]) {
    std::memcpy(m_fen, fen.c_str(), fen.size() + 1);
}

Fen& Fen::operator=(const std::string& fen) {
    if (m_fen != nullptr)
        delete[] m_fen;
    m_size = fen.size();
    m_fen  = new char[fen.size() + 1];
    std::memcpy(m_fen, fen.c_str(), fen.size() + 1);
    return *this;
}

Fen::~Fen() { delete[] m_fen; }

FenBatch::FenBatch(const std::vector<TrainingDataEntry>& entries) :
    m_size(entries.size()),
    m_fens(new Fen[entries.size()]) {
    for (int i = 0; i < m_size; ++i)
        m_fens[i] = entries[i].pos.fen();
}

FenBatch::~FenBatch() { delete[] m_fens; }

int FenBatchStream::calculate_num_reader_threads(int concurrency) {
    if (worker_thread_ratio >= 1)
        return 1;
    return std::max(1, concurrency - calculate_num_worker_threads(concurrency));
}

int FenBatchStream::calculate_num_worker_threads(int concurrency) {
    if (worker_thread_ratio <= 0)
        return 1;
    return std::max(1, static_cast<int>(std::floor(concurrency * worker_thread_ratio)));
}

FenBatchStream::FenBatchStream(int                                           concurrency,
                               const std::vector<std::string>&               filenames,
                               int                                           batch_size,
                               bool                                          cyclic,
                               std::function<bool(const TrainingDataEntry&)> skipPredicate,
                               int                                           rank,
                               int                                           world_size,
                               nnue::UniquePositionCounter*                  counter,
                               DataloaderIOConfig                            io_config) :
    BaseType(calculate_num_reader_threads(concurrency),
             filenames,
             cyclic,
             skipPredicate,
             counter,
             rank,
             world_size,
             io_config,
             calculate_num_worker_threads(concurrency)),
    m_batch_size(batch_size),
    m_concurrency(concurrency),
    m_num_workers(calculate_num_worker_threads(concurrency)),
    m_batch_queue_capacity(calculate_num_worker_threads(concurrency) + 4) {

    m_stop_flag.store(false);

    auto worker = [this]() {
        std::vector<TrainingDataEntry> entries;
        entries.reserve(m_batch_size);

        while (!m_stop_flag.load())
        {
            entries.clear();
            {
                BaseType::m_stream->fill_threadsafe(entries, m_batch_size);
                if (entries.empty())
                    break;
            }

            auto batch = new FenBatch(entries);

            {
                std::unique_lock lock(m_batch_mutex);
                m_batches_not_full.wait(lock, [this]() {
                    return m_batches.size() < static_cast<size_t>(m_batch_queue_capacity)
                        || m_stop_flag.load();
                });
                m_batches.emplace_back(batch);
                lock.unlock();
                m_batches_any.notify_one();
            }
        }
        m_num_workers.fetch_sub(1);
        m_batches_any.notify_one();
    };

    const int num_worker_threads = calculate_num_worker_threads(concurrency);
    for (int i = 0; i < num_worker_threads; ++i)
    {
        m_workers.emplace_back(worker);
    }
}

FenBatchStream::~FenBatchStream() {
    m_stop_flag.store(true);
    m_batches_not_full.notify_all();
    for (auto& worker : m_workers)
    {
        if (worker.joinable())
            worker.join();
    }
    for (auto& batch : m_batches)
        delete batch;
    // Counter is owned by AnyStream and deleted after m_stream (and its
    // workers) are destroyed, preventing use-after-free.
}

FenBatch* FenBatchStream::next() {
    std::unique_lock lock(m_batch_mutex);
    m_batches_any.wait(lock, [this]() { return !m_batches.empty() || m_num_workers.load() == 0; });
    if (!m_batches.empty())
    {
        auto batch = m_batches.front();
        m_batches.pop_front();
        lock.unlock();
        m_batches_not_full.notify_one();
        return batch;
    }
    return nullptr;
}

std::function<bool(const TrainingDataEntry&)> make_skip_predicate(DataloaderSkipConfig config) {
    if (!config.filtered && !config.wld_filtered && config.random_fen_skipping <= 0
        && config.early_fen_skipping < 0 && config.soft_early_fen_skipping <= 0)
    {
        return nullptr;
    }

    double   skip_prob             = 0.0;
    uint64_t random_skip_threshold = 0;
    if (config.random_fen_skipping > 0)
    {
        skip_prob = double(config.random_fen_skipping) / (config.random_fen_skipping + 1);
        random_skip_threshold = static_cast<uint64_t>(skip_prob * static_cast<double>(~0ULL));
    }

    // --- Precompute 5-Point Spline PC LUT ---
    std::array<double, 33> target_pc_weights_lut{};
    double                 target_pc_weights_total = 0.0;

    auto desired_piece_count_weights = [&config](int pc) -> double {
        double x    = static_cast<double>(pc);
        double y[5] = {config.pc_y0, config.pc_y1, config.pc_y2, config.pc_y3, config.pc_y4};

        if (x <= 0)
            return y[0];
        if (x >= 32)
            return y[4];

        int i = static_cast<int>(x / 8.0);
        if (i > 3)
            i = 3;

        double x0 = i * 8.0;
        double t  = (x - x0) / 8.0;

        auto get_slope = [&](int idx) {
            if (idx == 0)
                return (y[1] - y[0]);
            if (idx == 4)
                return (y[4] - y[3]);
            return (y[idx + 1] - y[idx - 1]) / 2.0;
        };

        double m0 = get_slope(i);
        double m1 = get_slope(i + 1);

        double t2 = t * t;
        double t3 = t2 * t;

        double h00 = 2 * t3 - 3 * t2 + 1;
        double h10 = t3 - 2 * t2 + t;
        double h01 = -2 * t3 + 3 * t2;
        double h11 = t3 - t2;

        double val = h00 * y[i] + h10 * m0 + h01 * y[i + 1] + h11 * m1;

        return std::max(0.0, val);
    };

    for (int i = 0; i < 33; ++i)
    {
        target_pc_weights_lut[i] = desired_piece_count_weights(i);
        target_pc_weights_total += target_pc_weights_lut[i];
    }
    if (target_pc_weights_total <= 0.0)
    {
        std::fill(target_pc_weights_lut.begin(), target_pc_weights_lut.end(), 1.0);
        target_pc_weights_total = static_cast<double>(target_pc_weights_lut.size());
    }

    // --- Precompute Soft Early Ply Filter LUT ---
    std::vector<double> early_ply_accept_prob;

    if (config.soft_early_fen_skipping > 0)
    {
        size_t lut_size = static_cast<size_t>(config.soft_early_fen_skipping) + 1;
        early_ply_accept_prob.resize(lut_size);

        auto interpolate_ply = [&config](double ply) -> double {
            struct Pt {
                double x, y;
            };
            Pt pts[5] = {{config.ply_x1, config.ply_y1},
                         {config.ply_x2, config.ply_y2},
                         {config.ply_x3, config.ply_y3},
                         {config.ply_x4, config.ply_y4},
                         {static_cast<double>(config.soft_early_fen_skipping), 1.0}};

            if (ply <= pts[0].x)
                return pts[0].y;
            if (ply >= pts[4].x)
                return pts[4].y;

            for (int i = 0; i < 4; ++i)
            {
                if (ply >= pts[i].x && ply <= pts[i + 1].x)
                {
                    if (pts[i + 1].x == pts[i].x)
                        return pts[i].y;
                    double t = (ply - pts[i].x) / (pts[i + 1].x - pts[i].x);
                    return pts[i].y + t * (pts[i + 1].y - pts[i].y);
                }
            }
            return 1.0;
        };

        for (size_t i = 0; i < lut_size; ++i)
        {
            early_ply_accept_prob[i] =
              std::clamp(interpolate_ply(static_cast<double>(i)), 0.0, 1.0);
        }
    }

    return [config, random_skip_threshold, target_pc_weights_lut, target_pc_weights_total,
            early_ply_accept_prob = std::move(early_ply_accept_prob)](const TrainingDataEntry& e) {
        static constexpr int    VALUE_NONE = 32002;
        static thread_local int last_ply   = -1;
        static thread_local int last_score = VALUE_NONE;

        // skip when score 0 was used as placeholder
        // detected through heuristic:
        // Game was not a draw and
        // last valid score was somewhat large

        bool skip_placeholder_zero = e.ply > last_ply && last_score != VALUE_NONE
                                  && std::abs(last_score) > 100 && e.result != 0 && e.score == 0;

        last_ply = e.ply;

        if (e.score == VALUE_NONE)
            return true;
        if (skip_placeholder_zero)
            return true;

        // Only update if valid score.
        last_score = e.score;

        // Hard Early Ply Filter
        if (e.ply <= config.early_fen_skipping)
            return true;

        auto& prng = rng::get_thread_local_rng();

        if (config.random_fen_skipping && (prng() < random_skip_threshold))
            return true;
        if (config.filtered && (e.isCapturingMove() || e.isInCheck()))
            return true;

        if (config.wld_filtered)
        {
            uint64_t wld_skip_threshold =
              static_cast<uint64_t>((1.0 - e.score_result_prob()) * static_cast<double>(~0ULL));
            if (prng() < wld_skip_threshold)
                return true;
        }

        if (config.simple_eval_skipping > 0
            && std::abs(e.pos.simple_eval()) < config.simple_eval_skipping)
        {
            return true;
        }

        // Soft Early Ply Filter
        if (config.soft_early_fen_skipping > 0 && e.ply < config.soft_early_fen_skipping)
        {
            uint64_t ply_reject_threshold = static_cast<uint64_t>(
              (1.0 - early_ply_accept_prob[e.ply]) * static_cast<double>(~0ULL));
            if (prng() < ply_reject_threshold)
            {
                return true;
            }
        }

        // Dynamic Piece Count Filter
        const int pc = e.pos.piecesBB().count();
        if (pc < 0 || pc > 32)
            return true;

        static thread_local double   alpha                   = 1.0;
        static thread_local double   pc_history_all[33]      = {0};
        static thread_local double   pc_history_passed[33]   = {0};
        static thread_local double   pc_history_all_total    = 0;
        static thread_local double   pc_history_passed_total = 0;
        static thread_local uint64_t step_count              = 0;

        const double max_pc_skip_rate = 0.975;

        pc_history_all[pc] += 1.0;
        pc_history_all_total += 1.0;
        step_count++;

        bool should_update =
          (step_count == 100 || step_count == 500 || step_count == 1000 || step_count == 2500
           || step_count == 5000 || (step_count > 5000 && step_count % 10000 == 0));

        if (should_update)
        {
            double min_ratio   = std::numeric_limits<double>::infinity();
            bool   found_valid = false;

            for (int i = 0; i < 33; ++i)
            {
                if (target_pc_weights_lut[i] > 0.0 && pc_history_all[i] > 0.0)
                {
                    double current_ratio = (pc_history_all_total * target_pc_weights_lut[i])
                                         / (target_pc_weights_total * pc_history_all[i]);
                    if (current_ratio < min_ratio)
                    {
                        min_ratio   = current_ratio;
                        found_valid = true;
                    }
                }
            }
            if (found_valid && min_ratio > 0.0)
            {
                alpha = (1.0 - max_pc_skip_rate) / min_ratio;
            }

            if (step_count >= 10000 && step_count % 10000 == 0)
            {
                for (int i = 0; i < 33; ++i)
                {
                    pc_history_all[i] *= 0.5;
                }
                pc_history_all_total *= 0.5;
            }
        }

        double accept_prob = 0.0;
        if (target_pc_weights_lut[pc] > 0.0 && pc_history_all[pc] > 0.0)
        {
            double current_ratio = (pc_history_all_total * target_pc_weights_lut[pc])
                                 / (target_pc_weights_total * pc_history_all[pc]);
            accept_prob          = alpha * current_ratio;
        }

        accept_prob = std::clamp(accept_prob, 0.0, 1.0);

        uint64_t reject_threshold =
          static_cast<uint64_t>((1.0 - accept_prob) * static_cast<double>(~0ULL));
        if (prng() < reject_threshold)
        {
            return true;
        }

        pc_history_passed[pc] += 1.0;
        pc_history_passed_total += 1.0;

        return false;
    };
}
