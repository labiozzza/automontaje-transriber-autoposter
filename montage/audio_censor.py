from __future__ import annotations

import re


_BANNED_WORDS_SOURCE = """
смерть умереть умер умерла мёртвый мёртвая труп убийство убийца убить убил убила
зарезать зарезал застрелить застрелил задушить расстрел казнь кровь кровавый
расчленение расчленить насилие изнасилование изнасиловать пытки похищение избить
драка суицид самоубийство повеситься повесился самоповреждение селфхарм спрыгнуть
передозировка передоз отравиться отравление яд утопиться утопился скончался
секс сексуальный сексуальная порно порнография эротика эротический эротическая
голый голая обнажённый обнажённая обнажёнка нюдсы интим интимный мастурбация
оргазм член пенис вагина сперма минет анальный проституция проститутка эскорт
сутенёр педофилия педофил домогательство
наркотики наркотик наркота кокаин героин метамфетамин мефедрон амфетамин экстази
LSD марихуана каннабис травка гашиш дилер закладка закладки наркотрафик
оружие пистолет автомат винтовка патроны боеприпасы бомба взрыв взрывчатка граната
стрельба нож терроризм террорист теракт
алкоголь водка виски пиво бухать бухло сигарета сигареты курить табак вейп
анорексия булимия голодание
мошенничество мошенник мошенники скам scam обман обмануть кража украсть взлом
взломать кардинг обнал обналичка отмывание казино ставки ставка букмекер азарт
пирамида Ponzi инсайд халява халявный бесплатно гарантированно гарантированный
гарантия безрисковый схема схемы обогащение скрыть обойти
деньги денег денежный денежная заработать зарабатывать заработок заработал
заработала доход доходы доходность прибыль прибыльный разбогатеть миллион миллионы
миллионер миллионеры инвестиции инвестировать инвестор вложения вложить вложение
кредит кредиты кредитка кредитный займ займы микрозайм долг долги должник банкротство
банкрот ипотека ипотечный банк банки банковский вклад вклады проценты процент ставка
зарплата зарплаты кешбэк кэшбэк рассрочка налог налоги обналичить обналичивание
вывести вывод перевод перевести пассивный капитал капитализация дивиденды дивиденд
акции акция крипта криптовалюта биткоин bitcoin трейдинг трейдер форекс депозит
окупаемость окупиться выгода выгодный бесплатный халява
"""

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")
_TIMING_RE = re.compile(
    r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*"
    r"(\d+):(\d+):(\d+)[,.](\d+)"
)


def normalize_word(value: str) -> str:
    return value.casefold().replace("ё", "е")


BANNED_WORDS = frozenset(normalize_word(word) for word in _BANNED_WORDS_SOURCE.split())


def _timestamp(parts: tuple[str, str, str, str]) -> float:
    hours, minutes, seconds, milliseconds = parts
    fraction = int(milliseconds[:3].ljust(3, "0")) / 1000.0
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + fraction


def needs_word_alignment(srt_text: str) -> bool:
    normalized_srt = srt_text.replace("\r\n", "\n").replace("\r", "\n")
    for block in re.split(r"\n\s*\n", normalized_srt.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        timing_index = next((index for index, line in enumerate(lines[:3]) if "-->" in line), None)
        if timing_index is None:
            continue
        tokens = _TOKEN_RE.findall(re.sub(r"<[^>]+>", "", " ".join(lines[timing_index + 1:])))
        if len(tokens) > 1 and any(normalize_word(token) in BANNED_WORDS for token in tokens):
            return True
    return False


def find_mute_intervals(srt_text: str) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    normalized_srt = srt_text.replace("\r\n", "\n").replace("\r", "\n")
    for block in re.split(r"\n\s*\n", normalized_srt.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        timing_index = next((index for index, line in enumerate(lines[:3]) if "-->" in line), None)
        if timing_index is None:
            continue
        match = _TIMING_RE.search(lines[timing_index])
        if not match:
            continue
        start = _timestamp(match.groups()[:4])
        end = _timestamp(match.groups()[4:])
        if end <= start:
            continue
        text = re.sub(r"<[^>]+>", "", " ".join(lines[timing_index + 1:]))
        tokens = _TOKEN_RE.findall(text)
        if not tokens:
            continue
        step = (end - start) / len(tokens)
        for index, token in enumerate(tokens):
            normalized = normalize_word(token)
            if normalized not in BANNED_WORDS:
                continue
            word_start = start + index * step
            word_end = word_start + step
            edge_fraction = 2 / len(normalized) if len(normalized) >= 6 else 0.25
            edge_fraction = max(0.12, min(0.35, edge_fraction))
            mute_start = word_start + step * edge_fraction
            mute_end = word_end - step * edge_fraction
            if mute_end - mute_start >= 0.02:
                intervals.append((mute_start, mute_end))

    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return [(round(start, 6), round(end, 6)) for start, end in merged]


def build_volume_filter(intervals: list[tuple[float, float]]) -> str:
    enabled = "+".join(
        f"between(t,{start:.6f},{end:.6f})"
        for start, end in intervals
        if end > start
    )
    return f"volume=0:enable='{enabled}'" if enabled else ""
