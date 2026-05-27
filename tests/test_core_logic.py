import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def reload_with_home(module_name: str, tmp_path: Path):
    module = importlib.import_module(module_name)
    module._FILE = tmp_path / f"{module_name}.json"
    module._cache = None
    module._cache_mtime = 0.0
    return module


@pytest.fixture
def fake_transcriber_deps(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(WhisperModel=object),
    )


def test_transcriber_postprocess_cleans_common_artifacts(fake_transcriber_deps):
    transcriber = importlib.import_module("transcriber")

    result = transcriber._postprocess("  , hej  hej ,du!!!  ")

    assert result == "Hej, du!"


def test_corrections_apply_case_insensitive_whole_words(tmp_path):
    corrections = reload_with_home("corrections", tmp_path)
    corrections.save({"motte": "möte"})

    # Capitalized source ("Motte") mirrors to capitalized replacement ("Möte"),
    # lowercase source stays lowercase. Inside-word substring "motteplats"
    # is left alone — word-boundary regex.
    result = corrections.apply("Motte idag, men motteplats ska vara kvar")

    assert result == "Möte idag, men motteplats ska vara kvar"


def test_corrections_apply_mirrors_all_caps(tmp_path):
    corrections = reload_with_home("corrections", tmp_path)
    corrections.save({"vinternote": "vintermöte"})

    result = corrections.apply("VINTERNOTE i morgon")

    assert result == "VINTERMÖTE i morgon"


def test_snippets_expand_exact_trigger_only(tmp_path):
    snippets = reload_with_home("snippets", tmp_path)
    snippets.save({"mvb": "Med vänliga hälsningar"})

    assert snippets.expand(" MVB ") == "Med vänliga hälsningar"
    assert snippets.expand("mvb tack") == "mvb tack"


def test_snippets_expand_strips_whisper_punctuation(tmp_path):
    snippets = reload_with_home("snippets", tmp_path)
    snippets.save({"mvb": "Med vänliga hälsningar"})

    # Whisper often appends a period or question mark to short utterances —
    # the snippet must still trigger.
    assert snippets.expand("MVB.") == "Med vänliga hälsningar"
    assert snippets.expand("mvb?") == "Med vänliga hälsningar"
    assert snippets.expand("Mvb!") == "Med vänliga hälsningar"
    assert snippets.expand("mvb…") == "Med vänliga hälsningar"


def test_snippets_expand_empty_lib_is_noop(tmp_path):
    snippets = reload_with_home("snippets", tmp_path)

    assert snippets.expand("hej") == "hej"


def test_auto_learn_extracts_same_length_word_diffs():
    auto_learn = importlib.import_module("auto_learn")

    diffs = auto_learn._extract_word_diffs(
        "Jag gar till motte.",
        "Jag går till möte.",
    )

    assert diffs == [("gar", "går"), ("motte", "möte")]


def test_auto_learn_majority_vote_wins_promotion(tmp_path, monkeypatch):
    """A single noisy LLM variant must not derail the dictionary."""
    auto_learn = importlib.reload(importlib.import_module("auto_learn"))
    monkeypatch.setattr(auto_learn, "LEARNED_FILE", tmp_path / "learned.json")
    monkeypatch.setattr(auto_learn, "PROMOTE_THRESHOLD", 3)

    promotions: list[tuple[str, str]] = []
    monkeypatch.setattr(auto_learn, "_promote",
                        lambda w, c: promotions.append((w, c)))

    # Three rounds: "möte" wins 2-1 over a one-time noisy "mode".
    auto_learn.record_correction("Jag gar till motte", "Jag går till möte")
    auto_learn.record_correction("Jag gar till motte", "Jag går till mode")
    auto_learn.record_correction("Jag gar till motte", "Jag går till möte")

    # "motte" must have promoted as the majority-vote winner.
    assert ("motte", "möte") in promotions
    learned = json.loads((tmp_path / "learned.json").read_text(encoding="utf-8"))
    assert learned["motte"]["correct"] == "möte"
    assert learned["motte"]["variants"] == {"möte": 2, "mode": 1}


def test_llm_polish_length_guard_allows_short_legitimate_polish(monkeypatch):
    """The old length guard rejected any polish < 30 % of input length,
    which threw away legitimate short polish on long input. The new guard
    only triggers when the result is BOTH < 30 % AND < 20 chars."""
    llm_polish = importlib.import_module("llm_polish")

    monkeypatch.setattr(llm_polish, "resolve_api_key", lambda k="": "fake")

    def fake_call(api_key, model, user_text, timeout_sec=8.0, system_prompt=""):
        # Simulate an LLM that returned a 22-char polish for a 100-char input.
        # That's 22 % — old guard would reject; new guard keeps it because
        # 22 chars is over the 20-char absolute floor.
        return {"choices": [{"message": {"content": "Det blir bra ändå idag."}}]}

    monkeypatch.setattr(llm_polish, "_call_api", fake_call)

    long_input = "öh så här liksom alltså jag tror egentligen att vi typ inte ska göra detta för det blir nog jättekonstigt"
    result = llm_polish.polish(long_input, "fake")

    # 23 chars < 30 % of 109 (=32.7) — old guard would have BLOCKED and
    # returned the original. New guard allows it.
    assert result.changed is True
    assert result.text == "Det blir bra ändå idag."


def test_llm_polish_length_guard_still_blocks_obvious_hallucinations(monkeypatch):
    """A 3-char polish on a 50-char input is still suspicious — keep it blocked."""
    llm_polish = importlib.import_module("llm_polish")

    monkeypatch.setattr(llm_polish, "resolve_api_key", lambda k="": "fake")
    monkeypatch.setattr(
        llm_polish, "_call_api",
        lambda *a, **k: {"choices": [{"message": {"content": "Ja."}}]},
    )

    long_input = "Detta är en mycket längre mening som inte borde bli till två tecken någonsin"
    result = llm_polish.polish(long_input, "fake")

    # 3 chars < 30 % of 76 AND 3 < 20 -> blocked, original returned.
    assert result.changed is False
    assert result.text == long_input


def test_corrections_apply_preserves_unmatched_text(tmp_path):
    corrections = reload_with_home("corrections", tmp_path)
    corrections.save({})

    assert corrections.apply("hej alla glada") == "hej alla glada"


def test_config_save_uses_keyring_and_excludes_secret(tmp_path, monkeypatch):
    config = importlib.import_module("config")
    secrets = {}

    config.CONFIG_DIR = tmp_path
    config.CONFIG_FILE = tmp_path / "config.json"
    config.keyring = SimpleNamespace(
        get_password=lambda service, username: secrets.get((service, username)),
        set_password=lambda service, username, value: secrets.__setitem__((service, username), value),
        delete_password=lambda service, username: secrets.pop((service, username), None),
    )

    config.save({**config.DEFAULTS, "llm_api_key": "secret-token", "model_size": "tiny"})

    saved = json.loads(config.CONFIG_FILE.read_text(encoding="utf-8"))
    loaded = config.load()

    assert "llm_api_key" not in saved
    assert loaded["llm_api_key"] == "secret-token"
    assert loaded["model_size"] == "tiny"


def test_config_load_migrates_legacy_secret_off_disk(tmp_path):
    config = importlib.import_module("config")
    secrets = {}

    config.CONFIG_DIR = tmp_path
    config.CONFIG_FILE = tmp_path / "config.json"
    config.CONFIG_FILE.write_text(
        json.dumps({"model_size": "base", "llm_api_key": "legacy-secret"}),
        encoding="utf-8",
    )
    config.keyring = SimpleNamespace(
        get_password=lambda service, username: secrets.get((service, username)),
        set_password=lambda service, username, value: secrets.__setitem__((service, username), value),
        delete_password=lambda service, username: secrets.pop((service, username), None),
    )

    loaded = config.load()
    saved = json.loads(config.CONFIG_FILE.read_text(encoding="utf-8"))

    assert loaded["llm_api_key"] == "legacy-secret"
    assert "llm_api_key" not in saved
    assert saved["model_size"] == "base"


def test_llm_polish_falls_back_without_logging_body(monkeypatch, caplog):
    llm_polish = importlib.import_module("llm_polish")

    def raise_http_error(*args, **kwargs):
        raise llm_polish.urllib.error.HTTPError(
            url="https://example.test",
            code=400,
            msg="bad request",
            hdrs=None,
            fp=SimpleNamespace(read=lambda: b"echoed sensitive text"),
        )

    monkeypatch.setattr(llm_polish, "_call_api", raise_http_error)

    with caplog.at_level("WARNING"):
        result = llm_polish.polish("hemlig text", "token")

    assert result.text == "hemlig text"
    assert not result.changed
    assert "hemlig text" not in caplog.text
    assert "echoed sensitive text" not in caplog.text


# --------------------------------------------------------------------------- #
#  Style preset (PR 2.1)                                                       #
# --------------------------------------------------------------------------- #

def test_style_default_falls_back_to_casual():
    llm_polish = importlib.import_module("llm_polish")
    prompt = llm_polish._build_system_prompt()
    assert llm_polish.STYLES["casual"] in prompt
    assert llm_polish._SYSTEM_PROMPT in prompt


def test_style_unknown_value_falls_back_to_default():
    llm_polish = importlib.import_module("llm_polish")
    prompt = llm_polish._build_system_prompt(style="nonsense-value")
    assert llm_polish.STYLES[llm_polish.DEFAULT_STYLE] in prompt


def test_style_each_preset_injects_its_guidance():
    llm_polish = importlib.import_module("llm_polish")
    for key, guidance in llm_polish.STYLES.items():
        prompt = llm_polish._build_system_prompt(style=key)
        assert guidance in prompt, f"style={key!r} did not include its guidance"


def test_style_custom_appends_user_prompt():
    llm_polish = importlib.import_module("llm_polish")
    prompt = llm_polish._build_system_prompt(
        style="custom",
        custom_prompt="Skriv alltid i tredje person.",
    )
    assert "Skriv alltid i tredje person." in prompt
    # Safety: the base prompt (which forbids content changes) is still there
    assert llm_polish._SYSTEM_PROMPT in prompt


def test_style_custom_empty_falls_back_to_default():
    llm_polish = importlib.import_module("llm_polish")
    prompt = llm_polish._build_system_prompt(style="custom", custom_prompt="   ")
    assert llm_polish.STYLES[llm_polish.DEFAULT_STYLE] in prompt


def test_polish_forwards_style_to_call_api(monkeypatch):
    """polish(style=..., custom_prompt=...) must reach _call_api as system_prompt."""
    llm_polish = importlib.import_module("llm_polish")
    monkeypatch.setattr(llm_polish, "resolve_api_key", lambda k="": "fake")

    captured = {}

    def fake_call(api_key, model, user_text, timeout_sec=8.0, system_prompt=""):
        captured["system_prompt"] = system_prompt
        return {"choices": [{"message": {"content": user_text}}]}

    monkeypatch.setattr(llm_polish, "_call_api", fake_call)

    llm_polish.polish("Hej världen.", "k", style="formal")
    assert llm_polish.STYLES["formal"] in captured["system_prompt"]


def test_style_in_config_defaults():
    config = importlib.import_module("config")
    assert config.DEFAULTS["style"] == "casual"
    assert config.DEFAULTS["custom_style_prompt"] == ""


def test_llm_polish_resolves_github_token_from_environment(monkeypatch):
    llm_polish = importlib.import_module("llm_polish")
    monkeypatch.setenv("GITHUB_TOKEN", "env-token")
    monkeypatch.delenv("GH_TOKEN", raising=False)

    assert llm_polish.resolve_api_key("") == "env-token"
    assert llm_polish.resolve_api_key("explicit-token") == "explicit-token"


def test_llm_polish_resolves_github_token_from_gh_cli(monkeypatch):
    llm_polish = importlib.import_module("llm_polish")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)

    def fake_run(*args, **kwargs):
        assert args[0] == ["gh", "auth", "token"]
        return SimpleNamespace(returncode=0, stdout="gh-token\n")

    monkeypatch.setattr(llm_polish.subprocess, "run", fake_run)
    assert llm_polish.resolve_api_key("") == "gh-token"


# --------------------------------------------------------------------------- #
#  New regression tests for refactors landed in this round.
# --------------------------------------------------------------------------- #

def test_corrections_apply_cache_invalidates_on_mtime(tmp_path, monkeypatch):
    """Editing the corrections file should invalidate the compiled-regex cache."""
    corrections = reload_with_home("corrections", tmp_path)
    corrections.save({"motte": "möte"})
    assert corrections.apply("motte") == "möte"

    # Save a new mapping; mtime advances so the cache must be rebuilt.
    # Force a strictly newer mtime to defeat low filesystem resolution.
    corrections.save({"gar": "går"})
    new_mtime = corrections.mtime() + 1
    import os
    os.utime(corrections._FILE, (new_mtime, new_mtime))

    assert corrections.apply("gar") == "går"
    # Old mapping no longer applies after replacement.
    assert corrections.apply("motte") == "motte"


def test_auto_learn_extracts_single_word_replacements_only():
    """Multi-word edits (insertions, deletions) must not yield bogus pairs."""
    auto_learn = importlib.import_module("auto_learn")

    # Length differs — naive zip would invent garbage; SequenceMatcher should
    # only emit the genuine single-word replace.
    diffs = auto_learn._extract_word_diffs(
        "Jag gar till skolan idag",
        "Jag går till skolan",
    )
    assert ("gar", "går") in diffs
    # The trailing deletion of "idag" must not appear as a replacement.
    assert all(b != "" for _, b in diffs)


def test_dictation_parse_hotkey_splits_modifiers_and_trigger():
    """_parse_hotkey must return ("space", ("ctrl", "shift")) for chorded hotkeys."""
    import sys as _sys
    from types import SimpleNamespace as _NS
    # Stub out the `keyboard` module so dictation imports cleanly in CI envs
    # without the real C extension installed.
    _sys.modules.setdefault(
        "keyboard",
        _NS(
            parse_hotkey=lambda s: ((1,), (2,)),  # tuple shape only matters
            is_pressed=lambda key: False,
            add_hotkey=lambda *a, **kw: None,
            on_release_key=lambda *a, **kw: None,
            unhook=lambda h: None,
            send=lambda *a, **kw: None,
        ),
    )
    _sys.modules.setdefault("sounds", _NS(play_start=lambda: None, play_stop=lambda: None))
    dictation = importlib.import_module("dictation")

    trigger, modifiers = dictation._parse_hotkey("ctrl+shift+space")
    assert trigger == "space"
    assert set(modifiers) == {"ctrl", "shift"}

    trigger, modifiers = dictation._parse_hotkey("f9")
    assert trigger == "f9"
    assert modifiers == ()


# --------------------------------------------------------------------------- #
#  Fas 1: modifier normalisation + dictation off-hook pipeline regression.
# --------------------------------------------------------------------------- #

def test_modifiers_normalize_aliases():
    """All aliases for the Windows key must collapse to 'windows'."""
    modifiers = importlib.import_module("modifiers")
    assert modifiers.normalize("win") == "windows"
    assert modifiers.normalize("Cmd") == "windows"
    assert modifiers.normalize("SUPER") == "windows"
    assert modifiers.normalize("control") == "ctrl"
    assert modifiers.normalize("ctrl") == "ctrl"
    assert modifiers.normalize("nonsense") is None
    assert modifiers.normalize("") is None


def test_modifiers_normalize_all_dedupes_and_preserves_order():
    modifiers = importlib.import_module("modifiers")
    result = modifiers.normalize_all(["Ctrl", "shift", "control", "win", "cmd"])
    # ctrl/control collapse; win/cmd collapse to windows.
    assert result == ("ctrl", "shift", "windows")


def test_modifiers_is_modifier():
    modifiers = importlib.import_module("modifiers")
    assert modifiers.is_modifier("alt")
    assert modifiers.is_modifier("CMD")
    assert not modifiers.is_modifier("space")


def test_dictation_parse_hotkey_normalises_cmd_to_windows():
    """A hotkey with 'cmd' must yield canonical 'windows' so paste releases it."""
    import sys as _sys
    from types import SimpleNamespace as _NS
    _sys.modules.setdefault(
        "keyboard",
        _NS(
            parse_hotkey=lambda s: ((1,), (2,)),
            is_pressed=lambda key: False,
            add_hotkey=lambda *a, **kw: None,
            on_press_key=lambda *a, **kw: None,
            on_release_key=lambda *a, **kw: None,
            unhook=lambda h: None,
            send=lambda *a, **kw: None,
            release=lambda k: None,
        ),
    )
    _sys.modules.setdefault("sounds", _NS(play_start=lambda: None,
                                          play_stop=lambda: None,
                                          play_error=lambda: None))
    dictation = importlib.import_module("dictation")

    trigger, modifiers = dictation._parse_hotkey("cmd+shift+space")
    assert trigger == "space"
    # 'cmd' must be normalised to 'windows' so paste._release_modifiers
    # actually releases the held Win key after Ctrl+V.
    assert set(modifiers) == {"windows", "shift"}


def test_audio_finalize_handles_empty_input():
    """finalize_audio must gracefully return an empty array for zero input."""
    import numpy as np
    audio = importlib.import_module("audio")
    result = audio.finalize_audio(np.empty(0, dtype=np.float32), 1, 16000)
    assert result.shape == (0,)
    assert result.dtype.name == "float32"


def test_audio_finalize_selects_loudest_stereo_channel():
    """Multi-channel input must preserve signal even if it is not on channel 0."""
    import numpy as np
    audio = importlib.import_module("audio")
    # Stereo buffer: left channel silent, right channel contains the mic.
    stereo = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
                      dtype=np.float32)
    result = audio.finalize_audio(stereo, 2, audio.TARGET_RATE)
    assert result.shape == (4,)
    assert np.allclose(result, 1.0)


def test_audio_finalize_averages_balanced_stereo_channels():
    import numpy as np
    audio = importlib.import_module("audio")
    stereo = np.array([[1.0, 0.5], [1.0, 0.5]], dtype=np.float32)

    result = audio.finalize_audio(stereo, 2, audio.TARGET_RATE)

    assert np.allclose(result, 0.75)


def test_audio_callback_rms_uses_loudest_channel(monkeypatch):
    """Silence gate RMS must not miss a mic signal on the right channel."""
    import numpy as np
    audio = importlib.import_module("audio")

    recorder = audio.MicRecorder()
    recorder._ensure_buffer(audio.TARGET_RATE, 2)
    recorder.recording = True
    indata = np.array([[0.0, 0.5], [0.0, 0.5]], dtype=np.float32)

    recorder._cb(indata, len(indata), None, None)

    assert recorder.level == pytest.approx(0.5)
    assert recorder.rms() == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
#  stop_fast_async (PR 2.10)                                                   #
# --------------------------------------------------------------------------- #

def test_stop_fast_async_returns_before_stream_close_completes():
    """Hook thread must not wait for PortAudio teardown.

    Simulates a slow stream.stop() (200 ms) and asserts that
    stop_fast_async returns in well under that — proves the close runs
    on the background thread, not inline.
    """
    import time as _time

    import numpy as np

    audio = importlib.import_module("audio")
    recorder = audio.MicRecorder()
    recorder._ensure_buffer(audio.TARGET_RATE, 1)
    recorder._buffer_offset = 5
    recorder._buffer[:5] = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    recorder._rate = audio.TARGET_RATE

    stop_calls = []

    class SlowStream:
        def stop(self):
            stop_calls.append("stop")
            _time.sleep(0.2)

        def close(self):
            stop_calls.append("close")

    recorder._stream = SlowStream()
    recorder.recording = True

    t0 = _time.perf_counter()
    captured, channels, rate = recorder.stop_fast_async()
    elapsed = _time.perf_counter() - t0

    # Caller path should be roughly the np.copy cost, not 200 ms.
    assert elapsed < 0.1, f"stop_fast_async blocked for {elapsed*1000:.0f} ms"
    assert captured.tolist() == [pytest.approx(x) for x in [0.1, 0.2, 0.3, 0.4, 0.5]]
    assert recorder._stream is None
    assert recorder.recording is False
    assert rate == audio.TARGET_RATE

    # The background thread must eventually finish both calls.
    assert recorder._pending_close is not None
    recorder._pending_close.join(timeout=1.0)
    assert stop_calls == ["stop", "close"]


def test_start_joins_pending_close_before_opening_new_stream(monkeypatch):
    """A fresh start() must wait for the previous async close to finish."""
    import time as _time

    audio = importlib.import_module("audio")
    recorder = audio.MicRecorder()

    join_observed = {"alive_at_start_call": None}

    class SlowStream:
        def stop(self):
            _time.sleep(0.15)

        def close(self):
            pass

    recorder._stream = SlowStream()
    recorder.recording = True
    recorder.stop_fast_async()
    pending = recorder._pending_close
    assert pending is not None
    assert pending.is_alive()

    # Force start() into a no-op for the actual PortAudio call so we only
    # measure the join behavior — patch _try_start to raise after the join.
    def _explode(*a, **kw):
        join_observed["alive_at_start_call"] = pending.is_alive()
        raise RuntimeError("stop here, we just want to verify the join")

    monkeypatch.setattr(audio, "_try_start", _explode)
    monkeypatch.setattr(recorder, "_build_candidates", lambda: [(0, 16000, 1, "x")])

    try:
        recorder.start()
    except Exception:
        pass

    # By the time start() reached _try_start, the pending close must have
    # been joined (i.e. the thread is no longer alive).
    assert join_observed["alive_at_start_call"] is False
    assert recorder._pending_close is None


def test_stop_fast_async_no_stream_does_not_spawn_thread():
    audio = importlib.import_module("audio")
    recorder = audio.MicRecorder()
    recorder._stream = None
    recorder._buffer_offset = 0

    captured, _channels, _rate = recorder.stop_fast_async()

    assert captured.size == 0
    assert recorder._pending_close is None


def test_corrections_apply_master_regex_handles_many_entries(tmp_path):
    """Master-regex path must apply all corrections in a single pass."""
    corrections = reload_with_home("corrections", tmp_path)
    mapping = {
        "motte": "möte",
        "gar": "går",
        "fika rasten": "fikarasten",
    }
    corrections.save(mapping)
    # Longest key first ensures multi-word "fika rasten" wins over individual
    # words that might overlap.
    result = corrections.apply("Jag gar pa motte under fika rasten idag")
    assert result == "Jag går pa möte under fikarasten idag"


def test_corrections_apply_empty_dictionary_is_noop(tmp_path):
    corrections = reload_with_home("corrections", tmp_path)
    corrections.save({})
    assert corrections.apply("oförändrad text") == "oförändrad text"


def test_indicator_push_level_throttles_redraws(monkeypatch):
    """push_level must coalesce rapid audio-thread calls into ≤1 pending
    Tk redraw — otherwise a 50 Hz audio callback floods after()."""
    # Stub out tkinter — we only need the FloatingIndicator class itself,
    # not a real Tk root.
    import sys as _sys
    fake_tk = type(_sys)("tkinter")
    fake_tk.Tk = object
    fake_tk.Toplevel = object
    fake_tk.Label = object
    fake_tk.Canvas = object
    fake_tk.Frame = object
    fake_tk.BooleanVar = object
    fake_tk.StringVar = object
    fake_tk.Button = object
    fake_tk.Entry = object
    fake_ttk = type(_sys)("tkinter.ttk")
    fake_ttk.Style = object
    fake_ttk.Combobox = object
    fake_ttk.Treeview = object
    fake_ttk.Scrollbar = object
    fake_messagebox = type(_sys)("tkinter.messagebox")
    fake_messagebox.showerror = lambda *a, **k: None
    fake_messagebox.askokcancel = lambda *a, **k: True
    fake_tk.ttk = fake_ttk
    fake_tk.messagebox = fake_messagebox
    monkeypatch.setitem(_sys.modules, "tkinter", fake_tk)
    monkeypatch.setitem(_sys.modules, "tkinter.ttk", fake_ttk)
    monkeypatch.setitem(_sys.modules, "tkinter.messagebox", fake_messagebox)

    ui = importlib.reload(importlib.import_module("ui"))

    scheduled: list = []

    class FakeRoot:
        def after(self, delay, fn=None, *args):
            scheduled.append((delay, fn, args))
            return 1
        def after_cancel(self, _id):
            pass

    ind = ui.FloatingIndicator(FakeRoot())
    # Simulate a shown listen window without invoking real Tk.
    ind._win = object()
    ind._canvas = object()
    ind._state = "listen"

    # 50 rapid pushes should result in exactly one scheduled redraw
    # (subsequent ones coalesce while _pending_push is True).
    for _ in range(50):
        ind.push_level(0.2)
    assert len(scheduled) == 1


def test_main_apply_settings_serialised(monkeypatch):
    """_apply_settings must acquire _config_lock before delegating, so
    two concurrent Save clicks can't interleave config mutations."""
    import threading as _th

    pytest.importorskip("PIL")
    pytest.importorskip("pystray")
    main = importlib.reload(importlib.import_module("main"))

    assert isinstance(main._config_lock, type(_th.Lock()))

    holds = []

    def tracer(cfg):
        holds.append(main._config_lock.locked())
        return True

    monkeypatch.setattr(main, "_apply_settings_locked", tracer)
    main._apply_settings({"hotkey": "ctrl+space"})
    assert holds == [True]


def test_config_load_keeps_legacy_secret_when_keyring_migration_fails(tmp_path):
    config = importlib.import_module("config")
    config.CONFIG_DIR = tmp_path
    config.CONFIG_FILE = tmp_path / "config.json"
    config.CONFIG_FILE.write_text(
        json.dumps({"model_size": "base", "llm_api_key": "legacy-secret"}),
        encoding="utf-8",
    )
    config.keyring = SimpleNamespace(
        get_password=lambda service, username: None,
        set_password=lambda service, username, value: (_ for _ in ()).throw(RuntimeError("no backend")),
        delete_password=lambda service, username: None,
    )

    loaded = config.load()
    saved = json.loads(config.CONFIG_FILE.read_text(encoding="utf-8"))

    assert loaded["llm_api_key"] == "legacy-secret"
    assert saved["llm_api_key"] == "legacy-secret"


def test_config_save_restores_secret_if_json_write_fails(tmp_path, monkeypatch):
    config = importlib.import_module("config")
    secrets = {(config._KEYRING_SERVICE, config._KEYRING_USERNAME): "old-secret"}
    config.CONFIG_DIR = tmp_path
    config.CONFIG_FILE = tmp_path / "config.json"
    config.keyring = SimpleNamespace(
        get_password=lambda service, username: secrets.get((service, username)),
        set_password=lambda service, username, value: secrets.__setitem__((service, username), value),
        delete_password=lambda service, username: secrets.pop((service, username), None),
    )
    monkeypatch.setattr(config, "save_json_atomic", lambda path, data: (_ for _ in ()).throw(RuntimeError("disk full")))

    with pytest.raises(RuntimeError):
        config.save({**config.DEFAULTS, "llm_api_key": "new-secret"})

    assert secrets[(config._KEYRING_SERVICE, config._KEYRING_USERNAME)] == "old-secret"


def test_config_save_fails_if_secret_delete_fails(tmp_path):
    config = importlib.import_module("config")
    secrets = {(config._KEYRING_SERVICE, config._KEYRING_USERNAME): "old-secret"}
    config.CONFIG_DIR = tmp_path
    config.CONFIG_FILE = tmp_path / "config.json"

    def delete_fail(service, username):
        raise RuntimeError("delete failed")

    config.keyring = SimpleNamespace(
        get_password=lambda service, username: secrets.get((service, username)),
        set_password=lambda service, username, value: secrets.__setitem__((service, username), value),
        delete_password=delete_fail,
    )

    with pytest.raises(RuntimeError):
        config.save({**config.DEFAULTS, "llm_api_key": ""})

    assert secrets[(config._KEYRING_SERVICE, config._KEYRING_USERNAME)] == "old-secret"


def test_llm_only_save_failure_restores_transcriber_state(monkeypatch):
    pytest.importorskip("PIL")
    pytest.importorskip("pystray")
    main = importlib.reload(importlib.import_module("main"))
    old_state = {
        "hotkey": "ctrl+space",
        "model_size": "small",
        "use_cuda": False,
        "llm_enabled": False,
        "llm_privacy_accepted": False,
        "llm_api_key": "old-key",
        "llm_model": "old-model",
    }
    main._config = old_state.copy()
    main._transcriber = SimpleNamespace(
        llm_enabled=False,
        llm_api_key="old-key",
        llm_model="old-model",
    )
    restarted = []
    monkeypatch.setattr(main, "_restart_dictation", lambda: restarted.append(True))
    monkeypatch.setattr(main.cfg_module, "save", lambda cfg: (_ for _ in ()).throw(RuntimeError("disk full")))

    result = main._apply_settings({
        "llm_enabled": True,
        "llm_privacy_accepted": True,
        "llm_api_key": "new-key",
        "llm_model": "new-model",
    })

    assert result is False
    assert main._config == old_state
    assert main._transcriber.llm_enabled is False
    assert main._transcriber.llm_api_key == "old-key"
    assert main._transcriber.llm_model == "old-model"
    assert len(restarted) >= 2


def test_paste_text_serializes_clipboard_workers(monkeypatch):
    paste = importlib.reload(importlib.import_module("paste"))
    events = []
    clipboard = {"value": "orig"}

    def fake_paste():
        events.append("read")
        return clipboard["value"]

    def fake_copy(value):
        events.append(("copy", value))
        clipboard["value"] = value

    monkeypatch.setattr(paste.pyperclip, "paste", fake_paste)
    monkeypatch.setattr(paste.pyperclip, "copy", fake_copy)
    monkeypatch.setattr(paste.keyboard, "send", lambda key: events.append(("send", key)))
    monkeypatch.setattr(paste, "_release_modifiers", lambda mods=(): None)
    monkeypatch.setattr(paste.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(paste.threading.Thread, "start", lambda self: self._target(*self._args, **self._kwargs))
    # Force non-console window so the ctrl+v branch is exercised (CI runs
    # under a real cmd.exe console which would otherwise route to shift+insert).
    monkeypatch.setattr(paste, "_active_window_class", lambda: "Notepad")

    paste.paste_text("first")
    paste.paste_text("second")

    assert events == [
        "read", ("copy", "first "), ("send", "ctrl+v"), ("copy", "orig"),
        "read", ("copy", "second "), ("send", "ctrl+v"), ("copy", "orig"),
    ]


def test_paste_text_uses_shift_insert_for_console_windows(monkeypatch):
    paste = importlib.reload(importlib.import_module("paste"))
    sent = []
    monkeypatch.setattr(paste, "_active_window_class", lambda: "ConsoleWindowClass")
    monkeypatch.setattr(paste.pyperclip, "paste", lambda: "orig")
    monkeypatch.setattr(paste.pyperclip, "copy", lambda value: None)
    monkeypatch.setattr(paste.keyboard, "send", lambda key: sent.append(key))
    monkeypatch.setattr(paste, "_release_modifiers", lambda mods=(): None)
    monkeypatch.setattr(paste.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(paste.threading.Thread, "start", lambda self: self._target(*self._args, **self._kwargs))

    paste.paste_text("hej")

    assert sent == ["shift+insert"]


def test_dictation_worker_does_not_paste_after_stop(monkeypatch):
    import numpy as np
    dictation = importlib.reload(importlib.import_module("dictation"))

    pasted = []
    mode = object.__new__(dictation.DictationMode)
    mode.transcriber = SimpleNamespace(transcribe=lambda audio: "stale text")
    mode._worker_stop = __import__("threading").Event()
    mode._worker_stop.set()
    mode._active = False
    mode._modifier_keys = ()
    mode.hotkey = "ctrl+space"
    mode.indicator = None
    mode.on_status = lambda msg: None
    monkeypatch.setattr(dictation, "inject_text",
                        lambda text, active_modifiers=(), strategy="auto", paste_threshold=200: pasted.append(text))

    mode._transcribe(np.ones(16000, dtype=np.float32))

    assert pasted == []


def test_transcriber_close_waits_for_inflight_transcribe(fake_transcriber_deps):
    import threading

    import numpy as np
    sys.modules["torch"] = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    )
    transcriber = importlib.reload(importlib.import_module("transcriber"))

    entered = threading.Event()
    release = threading.Event()

    class FakeModel:
        def __init__(self):
            self.in_transcribe = False
            self.closed_during_transcribe = False
        def transcribe(self, *args, **kwargs):
            def segments():
                self.in_transcribe = True
                entered.set()
                try:
                    # close() must not set owner.model to None while this generator runs.
                    release.wait(timeout=2.0)
                    yield SimpleNamespace(text="hej")
                finally:
                    self.in_transcribe = False
            return segments(), SimpleNamespace()

    inst = object.__new__(transcriber.Transcriber)
    inst.model_size = "small"
    inst.language = "sv"
    inst.llm_enabled = False
    inst.llm_api_key = ""
    inst.llm_model = "gpt-4.1-nano"
    inst.model = FakeModel()
    inst._model_lock = __import__("threading").RLock()
    original_model = inst.model

    result_holder = {}
    transcribe_thread = threading.Thread(
        target=lambda: result_holder.setdefault(
            "result", inst.transcribe(np.ones(16000, dtype=np.float32))
        )
    )
    transcribe_thread.start()
    assert entered.wait(timeout=1.0)

    close_done = threading.Event()
    close_thread = threading.Thread(target=lambda: (inst.close(), close_done.set()))
    close_thread.start()

    assert not close_done.wait(timeout=0.05)
    release.set()
    transcribe_thread.join(timeout=1.0)
    close_thread.join(timeout=1.0)

    assert result_holder["result"] == "Hej"
    assert close_done.is_set()
    assert inst.model is None
    assert original_model.in_transcribe is False

# ---------- text_inject (PR 1.2 hybrid paste) ---------- #

def test_text_inject_short_text_uses_keyboard_write(monkeypatch):
    """Default threshold is 200 chars — short text bypasses clipboard."""
    text_inject = importlib.reload(importlib.import_module("text_inject"))

    typed = []
    # Some other tests have stubbed sys.modules["keyboard"] with a partial
    # namespace; ensure write exists on whatever stub text_inject sees.
    monkeypatch.setattr(text_inject.keyboard, "write",
                        lambda text, delay=0: typed.append(text),
                        raising=False)
    monkeypatch.setattr(text_inject, "_release_modifiers", lambda mods=(): None)
    monkeypatch.setattr(text_inject, "_inject_via_clipboard",
                        lambda text, gen: (_ for _ in ()).throw(
                            AssertionError("clipboard path must not run for short text")))

    ok = text_inject.inject("hej alla glada")

    assert ok is True
    assert typed == ["hej alla glada "]  # trailing space appended


def test_text_inject_long_text_uses_clipboard(monkeypatch):
    """Text longer than the threshold falls back to clipboard."""
    text_inject = importlib.reload(importlib.import_module("text_inject"))

    routed = []
    monkeypatch.setattr(text_inject, "_release_modifiers", lambda mods=(): None)
    monkeypatch.setattr(text_inject, "_inject_via_keyboard",
                        lambda text: (_ for _ in ()).throw(
                            AssertionError("keyboard path must not run for long text")))
    monkeypatch.setattr(text_inject, "_inject_via_clipboard",
                        lambda text, gen: routed.append(("clipboard", text, gen)) or True)

    long_text = "ord " * 80  # ~320 chars, well past 200 threshold
    ok = text_inject.inject(long_text)

    assert ok is True
    assert routed[0][0] == "clipboard"


def test_text_inject_forced_clipboard_strategy_ignores_threshold(monkeypatch):
    text_inject = importlib.reload(importlib.import_module("text_inject"))

    routed = []
    monkeypatch.setattr(text_inject, "_release_modifiers", lambda mods=(): None)
    monkeypatch.setattr(text_inject, "_inject_via_keyboard",
                        lambda text: (_ for _ in ()).throw(AssertionError("must not run")))
    monkeypatch.setattr(text_inject, "_inject_via_clipboard",
                        lambda text, gen: routed.append(text) or True)

    text_inject.inject("kort text", strategy="clipboard")
    assert routed == ["kort text"]


def test_text_inject_forced_inject_strategy_ignores_threshold(monkeypatch):
    text_inject = importlib.reload(importlib.import_module("text_inject"))

    typed = []
    monkeypatch.setattr(text_inject, "_release_modifiers", lambda mods=(): None)
    monkeypatch.setattr(text_inject.keyboard, "write",
                        lambda text, delay=0: typed.append(text),
                        raising=False)
    monkeypatch.setattr(text_inject, "_inject_via_clipboard",
                        lambda text, gen: (_ for _ in ()).throw(AssertionError("must not run")))

    long_text = "ord " * 80
    text_inject.inject(long_text, strategy="inject")

    assert typed and typed[0].startswith("ord ord ord")


def test_text_inject_keyboard_failure_falls_back_to_clipboard(monkeypatch):
    text_inject = importlib.reload(importlib.import_module("text_inject"))

    monkeypatch.setattr(text_inject, "_release_modifiers", lambda mods=(): None)

    def boom(text, delay=0):
        raise RuntimeError("layout missing char")
    monkeypatch.setattr(text_inject.keyboard, "write", boom, raising=False)

    routed = []
    monkeypatch.setattr(text_inject, "_inject_via_clipboard",
                        lambda text, gen: routed.append(text) or True)

    ok = text_inject.inject("hej")

    assert ok is True
    assert routed == ["hej"]


def test_text_inject_empty_text_is_noop(monkeypatch):
    text_inject = importlib.reload(importlib.import_module("text_inject"))

    monkeypatch.setattr(text_inject, "_release_modifiers",
                        lambda mods=(): (_ for _ in ()).throw(AssertionError("must not run")))

    assert text_inject.inject("   ") is False
    assert text_inject.inject("") is False


def test_text_inject_generation_counter_increments():
    text_inject = importlib.reload(importlib.import_module("text_inject"))

    g1 = text_inject._next_generation()
    g2 = text_inject._next_generation()
    g3 = text_inject._next_generation()

    assert g1 < g2 < g3
    assert text_inject._current_generation() == g3


def test_text_inject_restore_skips_when_newer_generation(monkeypatch):
    """A second dictation starting mid-restore must cancel the first restore."""
    text_inject = importlib.reload(importlib.import_module("text_inject"))

    restored = []
    monkeypatch.setattr(text_inject.pyperclip, "copy",
                        lambda v: restored.append(v))
    # Pretend the clipboard always still holds our dictated marker — would
    # otherwise loop until timeout.
    monkeypatch.setattr(text_inject.pyperclip, "paste", lambda: "dictated text ")
    monkeypatch.setattr(text_inject.time, "sleep", lambda s: None)
    monkeypatch.setattr(text_inject.time, "monotonic", lambda: 0.0)

    # gen=1 was issued, but generation has already advanced past it.
    text_inject._next_generation()  # gen = 1
    text_inject._next_generation()  # gen = 2 (newer)

    text_inject._restore_clipboard("old", "dictated text ", gen=1)

    # Restore must not run because gen 1 is stale.
    assert restored == []

# ---------- model_ui (PR 1.3) ---------- #

def test_model_ui_model_is_local_uses_find_local_model(monkeypatch, tmp_path):
    """model_is_local must mirror transcriber._find_local_model."""
    # Stub the heavy imports model_ui pulls (convert_model imports transformers
    # lazily; transcriber is safe but pulls faster_whisper). We reload with
    # fake modules in sys.modules so model_ui sees our stubs.
    import sys as _sys
    fake_convert = SimpleNamespace(
        KBLAB_MODELS={
            "tiny": "KBLab/kb-whisper-tiny",
            "small": "KBLab/kb-whisper-small",
            "large": "KBLab/kb-whisper-large",
        },
        convert=lambda size: None,
    )
    fake_transcriber = SimpleNamespace(
        MODEL_DIR=tmp_path / "models",
        _find_local_model=lambda repo: "/fake/path" if repo == "KBLab/kb-whisper-small" else None,
    )
    monkeypatch.setitem(_sys.modules, "convert_model", fake_convert)
    monkeypatch.setitem(_sys.modules, "transcriber", fake_transcriber)
    if "model_ui" in _sys.modules:
        del _sys.modules["model_ui"]
    model_ui = importlib.import_module("model_ui")

    assert model_ui.model_is_local("small") is True
    assert model_ui.model_is_local("large") is False
    assert model_ui.model_is_local("unknown-size") is False


def test_model_ui_delete_local_model_removes_existing_dirs(monkeypatch, tmp_path):
    import sys as _sys
    models_dir = tmp_path / "models"
    ct2 = models_dir / "kb-whisper-tiny-ct2"
    hf = models_dir / "models--KBLab--kb-whisper-tiny"
    ct2.mkdir(parents=True)
    (ct2 / "model.bin").write_bytes(b"x")
    hf.mkdir(parents=True)
    (hf / "config.json").write_text("{}")

    fake_convert = SimpleNamespace(
        KBLAB_MODELS={"tiny": "KBLab/kb-whisper-tiny"},
        convert=lambda size: None,
    )
    fake_transcriber = SimpleNamespace(
        MODEL_DIR=models_dir,
        _find_local_model=lambda repo: None,
    )
    monkeypatch.setitem(_sys.modules, "convert_model", fake_convert)
    monkeypatch.setitem(_sys.modules, "transcriber", fake_transcriber)
    if "model_ui" in _sys.modules:
        del _sys.modules["model_ui"]
    model_ui = importlib.import_module("model_ui")

    removed = model_ui.delete_local_model("tiny")

    assert removed is True
    assert not ct2.exists()
    assert not hf.exists()


def test_model_ui_delete_local_model_returns_false_when_nothing_to_remove(monkeypatch, tmp_path):
    import sys as _sys
    fake_convert = SimpleNamespace(
        KBLAB_MODELS={"tiny": "KBLab/kb-whisper-tiny"},
        convert=lambda size: None,
    )
    fake_transcriber = SimpleNamespace(
        MODEL_DIR=tmp_path / "missing",
        _find_local_model=lambda repo: None,
    )
    monkeypatch.setitem(_sys.modules, "convert_model", fake_convert)
    monkeypatch.setitem(_sys.modules, "transcriber", fake_transcriber)
    if "model_ui" in _sys.modules:
        del _sys.modules["model_ui"]
    model_ui = importlib.import_module("model_ui")

    assert model_ui.delete_local_model("tiny") is False

# ---------- privacy_ui (PR 1.4) ---------- #

def test_privacy_wipe_path_removes_file(tmp_path):
    import importlib
    privacy_ui = importlib.reload(importlib.import_module("privacy_ui"))
    f = tmp_path / "x.json"
    f.write_text("hello")
    assert privacy_ui.wipe_path(f) is True
    assert not f.exists()


def test_privacy_wipe_path_removes_directory(tmp_path):
    import importlib
    privacy_ui = importlib.reload(importlib.import_module("privacy_ui"))
    d = tmp_path / "models"
    (d / "sub").mkdir(parents=True)
    (d / "sub" / "a.bin").write_bytes(b"x")
    assert privacy_ui.wipe_path(d) is True
    assert not d.exists()


def test_privacy_wipe_path_returns_false_when_missing(tmp_path):
    import importlib
    privacy_ui = importlib.reload(importlib.import_module("privacy_ui"))
    assert privacy_ui.wipe_path(tmp_path / "does-not-exist") is False


def test_privacy_path_size_human_handles_missing(tmp_path):
    import importlib
    privacy_ui = importlib.reload(importlib.import_module("privacy_ui"))
    assert privacy_ui._path_size_human(tmp_path / "missing") == "— saknas"


def test_privacy_path_size_human_file(tmp_path):
    import importlib
    privacy_ui = importlib.reload(importlib.import_module("privacy_ui"))
    f = tmp_path / "x.bin"
    f.write_bytes(b"x" * 2048)  # 2 KB
    out = privacy_ui._path_size_human(f)
    assert "KB" in out or "MB" in out  # 2048 bytes -> "2.0 KB"


def test_privacy_path_size_human_directory(tmp_path):
    import importlib
    privacy_ui = importlib.reload(importlib.import_module("privacy_ui"))
    d = tmp_path / "dir"
    d.mkdir()
    (d / "a").write_bytes(b"x" * 1000)
    (d / "b").write_bytes(b"x" * 1000)
    out = privacy_ui._path_size_human(d)
    # 2000 bytes total; should report KB or similar, not '— saknas'
    assert "saknas" not in out
    assert any(unit in out for unit in ("B", "KB", "MB"))


def test_privacy_data_targets_uses_current_config_dir(monkeypatch, tmp_path):
    """_data_targets() resolves CONFIG_DIR at call time, not import time."""
    import sys as _sys
    fake_cfg = SimpleNamespace(CONFIG_DIR=tmp_path / "fakehome")
    monkeypatch.setitem(_sys.modules, "config", fake_cfg)
    if "privacy_ui" in _sys.modules:
        del _sys.modules["privacy_ui"]
    privacy_ui = importlib.import_module("privacy_ui")
    targets = privacy_ui._data_targets()
    log_path, _desc = targets["Loggfil"]
    assert log_path == tmp_path / "fakehome" / "freewispr.log"
