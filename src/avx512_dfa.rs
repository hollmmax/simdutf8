//! An UTF-8 validation implemented using a DFA.
//! Hoewver rather than interpreting the DFA conventionally, the input is converted to
//! State->State transitions which can then be reduced associatively.

use core::{arch::x86_64::*, mem::transmute};

static C_LUT: [u8; 128] = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 7, 7, 9, 10, 10, 10, 11, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12,
];
static T_LUT: [u64; 16] = [
    0x06060606060600,
    0x06060604040201,
    0x06060605050201,
    0x06060605050301,
    0x06060606060606,
    0x06060606060006,
    0x06060600060606,
    0x06060600000606,
    0x06060606000606,
    0x06000606060606,
    0x06000006060606,
    0x06060006060606,
    0x06060606060606,
    0x00000000000000,
    0x00000000000000,
    0x00000000000000,
];

/// # Safety
/// Lul no :kekwyou:
#[unsafe(no_mangle)]
pub fn check_utf8(data: &[u8]) -> bool {
    unsafe {
        let c_lut1 = _mm512_loadu_epi8(&C_LUT[0] as *const u8 as *const i8);
        let c_lut2 = _mm512_loadu_epi8(&C_LUT[64] as *const u8 as *const i8);
        let t_lut1 = _mm512_loadu_epi64(&T_LUT[0] as *const u64 as *const i64);
        let t_lut2 = _mm512_loadu_epi64(&T_LUT[8] as *const u64 as *const i64);
        let shuffle_offset = _mm512_maskz_set1_epi32(_cvtu32_mask16(0xcccc), 0x08080808);

        #[rustfmt::skip]
        let process_chunk = |chunk: &[u8; 64]| {
            let chunk = _mm512_loadu_epi8(chunk.as_ptr() as *const i8);
            let mask = _mm512_movepi8_mask(chunk);
            let classes = _mm512_maskz_permutex2var_epi8(mask, c_lut1, chunk, c_lut2);
            let transitions_0 = _mm512_permutex2var_epi64(t_lut1, classes, t_lut2);
            let transitions_1 = _mm512_permutex2var_epi64(t_lut1, _mm512_srli_epi64(classes, 8), t_lut2);
            let transitions_2 = _mm512_permutex2var_epi64(t_lut1, _mm512_srli_epi64(classes, 16), t_lut2);
            let transitions_3 = _mm512_permutex2var_epi64(t_lut1, _mm512_srli_epi64(classes, 24), t_lut2);
            let transitions_4 = _mm512_permutex2var_epi64(t_lut1, _mm512_srli_epi64(classes, 32), t_lut2);
            let transitions_5 = _mm512_permutex2var_epi64(t_lut1, _mm512_srli_epi64(classes, 40), t_lut2);
            let transitions_6 = _mm512_permutex2var_epi64(t_lut1, _mm512_srli_epi64(classes, 48), t_lut2);
            let transitions_7 = _mm512_permutex2var_epi64(t_lut1, _mm512_srli_epi64(classes, 56), t_lut2);
            let transitions_01 = _mm512_shuffle_epi8(transitions_0, _mm512_or_epi64(transitions_1, shuffle_offset));
            let transitions_23 = _mm512_shuffle_epi8(transitions_2, _mm512_or_epi64(transitions_3, shuffle_offset));
            let transitions_45 = _mm512_shuffle_epi8(transitions_4, _mm512_or_epi64(transitions_5, shuffle_offset));
            let transitions_67 = _mm512_shuffle_epi8(transitions_6, _mm512_or_epi64(transitions_7, shuffle_offset));
            let transitions_0123 = _mm512_shuffle_epi8(transitions_01, _mm512_or_epi64(transitions_23, shuffle_offset));
            let transitions_4567 = _mm512_shuffle_epi8(transitions_45, _mm512_or_epi64(transitions_67, shuffle_offset));
            _mm512_shuffle_epi8(transitions_0123, _mm512_or_epi64(transitions_4567, shuffle_offset))
        };

        // NOTE: Merging can be done using 2 shuffles to perform a horizontal reduction in each
        // vector separately and then a permute to join them together. This should result in lower
        // latency on most CPUs, but likely worse throughput.
        let merge_shuffle = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
        let merge_mask = _mm512_set_epi64(
            0x7070707070707070u64 as i64,
            0x6060606060606060u64 as i64,
            0x5050505050505050u64 as i64,
            0x4040404040404040u64 as i64,
            0x3030303030303030u64 as i64,
            0x2020202020202020u64 as i64,
            0x1010101010101010u64 as i64,
            0x0000000000000000u64 as i64,
        );
        let merge_adjecent = |a: __m512i, b: __m512i| {
            let idx = _mm512_permutex2var_epi64(a, merge_shuffle, b);
            let idx = _mm512_or_epi64(idx, merge_mask);
            _mm512_permutex2var_epi8(a, idx, b)
        };

        let (bulk, remainder) = data.as_chunks::<64>();
        let transition = bulk
            .iter()
            .map(process_chunk)
            .fold(_mm512_set1_epi64(0x0706050403020100), |a, b| {
                merge_adjecent(a, b)
            });
        remainder
            .iter()
            .rev()
            .map(|c| {
                *T_LUT.get_unchecked(if *c < 128 {
                    0
                } else {
                    C_LUT[(*c - 128) as usize]
                } as usize)
            })
            .chain(
                transmute::<__m512i, [u64; 8]>(transition)
                    .iter()
                    .copied()
                    .rev(),
            )
            .fold(0, |acc, s| s.to_le_bytes()[acc as usize])
            == 0
    }
}
