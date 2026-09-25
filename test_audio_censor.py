import unittest

from montage.audio_censor import (
    BANNED_WORDS,
    build_volume_filter,
    find_mute_intervals,
    needs_word_alignment,
)


class AudioCensorTests(unittest.TestCase):
    def test_matches_complete_words_case_and_yo_variants(self):
        srt = """1
00:00:00,000 --> 00:00:03,000
ДЕНЬГИ, мёртвый и scam.
"""
        intervals = find_mute_intervals(srt)
        self.assertEqual(len(intervals), 3)
        self.assertEqual(intervals[0], (0.25, 0.5))

    def test_does_not_match_substrings(self):
        srt = """1
00:00:00,000 --> 00:00:02,000
банкир переводчик
"""
        self.assertEqual(find_mute_intervals(srt), [])

    def test_splits_phrase_timing_and_keeps_word_edges(self):
        srt = """1
00:00:10,000 --> 00:00:12,000
мои деньги сегодня
"""
        self.assertEqual(find_mute_intervals(srt), [(10.888889, 11.111111)])

    def test_builds_single_ffmpeg_enable_expression(self):
        audio_filter = build_volume_filter([(0.25, 0.75), (2.0, 2.5)])
        self.assertEqual(
            audio_filter,
            "volume=0:enable='between(t,0.250000,0.750000)+between(t,2.000000,2.500000)'",
        )
        self.assertEqual(build_volume_filter([]), "")

    def test_required_english_and_russian_words_are_present(self):
        for word in (
            "убийство", "деньги", "заработок", "мертвый", "наркотики",
            "lsd", "scam", "ponzi", "bitcoin",
        ):
            self.assertIn(word, BANNED_WORDS)

    def test_mutes_reported_sensitive_words(self):
        srt = """1
00:00:00,000 --> 00:00:01,000
убийство

2
00:00:01,000 --> 00:00:02,000
деньги

3
00:00:02,000 --> 00:00:03,000
заработок
"""
        self.assertEqual(
            find_mute_intervals(srt),
            [(0.25, 0.75), (1.333333, 1.666667), (2.222222, 2.777778)],
        )

    def test_requests_alignment_for_phrase_level_sensitive_words(self):
        phrase_srt = """1
00:00:00,000 --> 00:00:06,660
Чувак за деньги совершил убийство
"""
        word_srt = """1
00:00:00,760 --> 00:00:01,240
деньги
"""
        self.assertTrue(needs_word_alignment(phrase_srt))
        self.assertFalse(needs_word_alignment(word_srt))


if __name__ == "__main__":
    unittest.main()
