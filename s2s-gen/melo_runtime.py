"""MeloTTS 실행 전 MeCab/UniDic 런타임 보정."""
from __future__ import annotations

import os
import sys
import types
import importlib.machinery
from pathlib import Path


def prepare_unidic_lite() -> None:
    """MeloTTS import 전에 mecab-python3 이 unidic-lite 사전을 보도록 설정한다.

    MeloTTS 는 한국어만 써도 import 시 일본어 cleaner 를 함께 로드하고, 그 과정에서
    MeCab.Tagger() 가 기본 unidic 사전의 mecabrc 를 찾는다. `uv --with` 임시 환경에서는
    unidic 패키지는 있어도 `python -m unidic download` 데이터가 없어 실패하므로, 패키지에
    사전이 포함된 unidic-lite 를 명시적으로 가리킨다.
    """
    try:
        import unidic_lite  # type: ignore
    except Exception:
        return

    dicdir = Path(unidic_lite.DICDIR)
    mecabrc = dicdir / "mecabrc"
    if mecabrc.exists():
        os.environ.setdefault("MECABRC", str(mecabrc))

    try:
        import unidic  # type: ignore

        unidic.DICDIR = str(dicdir)
    except Exception:
        pass

    prepare_korean_mecab_adapter()


def prepare_korean_mecab_adapter() -> None:
    """Make MeloTTS Korean G2P work when mecab-python3 and python-mecab-ko collide.

    Melo imports Japanese cleaners at module import time and expects `MeCab.Tagger`.
    Korean g2pkk later expects `mecab.MeCab().pos()`. In a `uv --with melotts
    --with python-mecab-ko` environment both distributions can overlap on the
    lowercase `mecab` package, leaving Tagger available but MeCab absent. This
    adapter supplies the python-mecab-ko style `pos()` API using mecab-python3
    with python-mecab-ko-dic.
    """
    try:
        import mecab  # type: ignore
    except Exception:
        mecab = types.ModuleType("mecab")  # type: ignore
        mecab.__spec__ = importlib.machinery.ModuleSpec("mecab", loader=None)
        sys.modules["mecab"] = mecab

    if "MeCab" not in sys.modules:
        shim = types.ModuleType("MeCab")

        class _DummyTagger:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def parse(self, text: str = "") -> str:
                return ""

        shim.Tagger = getattr(mecab, "Tagger", _DummyTagger)
        shim.__spec__ = importlib.machinery.ModuleSpec("MeCab", loader=None)
        sys.modules["MeCab"] = shim

    if hasattr(mecab, "MeCab"):
        return

    try:
        import mecab_ko_dic  # type: ignore
    except Exception:
        return

    class _KoreanMeCab:
        def __init__(self) -> None:
            self._tagger = None
            if hasattr(mecab, "Tagger"):
                self._tagger = mecab.Tagger(f"-r /dev/null -d {mecab_ko_dic.dictionary_path}")

        def pos(self, text: str):
            if self._tagger is None:
                return [(token, "NNG") for token in text.split() if token]
            rows = []
            for line in (self._tagger.parse(text) or "").splitlines():
                if not line or line == "EOS" or "\t" not in line:
                    continue
                surface, features = line.split("\t", 1)
                tag = features.split(",", 1)[0]
                rows.append((surface, tag))
            return rows

    mecab.MeCab = _KoreanMeCab
