"""Procedural sound for Belt Fighter.

All sounds are synthesized at startup — no audio assets, no files.
SoundBank is a pure side-effect layer: the sim never reads from it and
never feeds it rng, so determinism (test_determinism.py) is unaffected.

S2: full recipe set. Synthesis uses numpy when available; without it the
bank degrades to a single placeholder tone (game still runs, mostly silent).
"""
import array
import math

import pygame

try:
    import numpy as _np
except ImportError:
    _np = None

from .config import SFX_MASTER_VOLUME, SFX_VOLUMES

MIX_RATE = 44100


def _to_sound(samples, rate):
    """Float samples in [-1, 1] -> pygame Sound (16-bit mono)."""
    if _np is not None:
        arr = _np.clip(_np.asarray(samples, dtype=_np.float32), -1.0, 1.0)
        arr = (arr * 32767).astype(_np.int16)
        return pygame.sndarray.make_sound(arr)
    # No numpy: build the PCM buffer by hand.
    pcm = array.array("h")
    for s in samples:
        pcm.append(int(max(-1.0, min(1.0, s)) * 32767))
    return pygame.mixer.Sound(buffer=pcm.tobytes())


class SoundBank:
    """Holds synthesized sounds and plays them.

    Created in __main__ after pygame.init(); passed into Game as an
    optional dependency (None = silent, e.g. headless/test runs).
    """

    def __init__(self):
        self.ready = False
        self.muted = False
        self.rate = MIX_RATE
        self._sounds = {}
        self._thruster_on = False
        self._thruster_channel = None
        self._last_play = {}

    def init(self):
        """Init the mixer (mono 16-bit) and synthesize sounds.

        Fails soft: with no audio device the game just runs silent.
        """
        if self.ready:
            return
        try:
            pygame.mixer.quit()
            pygame.mixer.init(MIX_RATE, -16, 1, 512)
            self.rate = pygame.mixer.get_init()[0]
        except pygame.error:
            return
        if _np is None:
            # No numpy: keep the bank alive with just the placeholder.
            print("sound: numpy not found — running with placeholder audio only")
            self._sounds["placeholder"] = self._tone(440.0, 0.15)
            self.ready = True
            return
        self._build_all()
        self.ready = True

    # --- public API ----------------------------------------------------
    def play(self, name):
        """Fire-and-forget. Unknown names are ignored (forward-compat)."""
        if not self.ready or self.muted:
            return
        snd = self._sounds.get(name)
        if snd is not None:
            snd.play()

    def play_throttled(self, name, min_interval):
        """Play at most once per min_interval seconds (wall clock).

        For rapid-fire weapons: the sim fires ~100 shots/s but the ear
        wants ~8 blips/s. Throttle state lives here, not in Game, so the
        sim stays sound-free.
        """
        if not self.ready or self.muted:
            return
        now = pygame.time.get_ticks()
        if now - self._last_play.get(name, -10**9) < min_interval * 1000.0:
            return
        self._last_play[name] = now
        snd = self._sounds.get(name)
        if snd is not None:
            snd.play()

    def set_thruster(self, on):
        """Engine loop on/off. Self-heals if another sound stole the
        channel (get_sound() is None -> restart the loop)."""
        if not self.ready or "thruster" not in self._sounds:
            return
        if on:
            ch = self._thruster_channel
            if ch is None or ch.get_sound() is None:
                ch = self._sounds["thruster"].play(loops=-1)
                self._thruster_channel = ch
            self._thruster_on = True
        else:
            if self._thruster_channel is not None:
                self._thruster_channel.stop()
                self._thruster_channel = None
            self._thruster_on = False

    def mute(self, m):
        self.muted = bool(m)
        if m and self._thruster_channel is not None:
            self._thruster_channel.stop()
            self._thruster_channel = None
            self._thruster_on = False

    # --- synthesis: shared helpers --------------------------------------
    def _vol(self, name):
        return SFX_MASTER_VOLUME * SFX_VOLUMES.get(name, 0.5)

    def _reg(self, name, samples):
        snd = _to_sound(samples, self.rate)
        snd.set_volume(self._vol(name))
        self._sounds[name] = snd

    def _t(self, dur):
        return _np.arange(int(self.rate * dur)) / self.rate

    def _sweep(self, f0, f1, dur):
        """Sine with a linear frequency ramp f0 -> f1 (phase-integrated)."""
        t = self._t(dur)
        phase = 2.0 * math.pi * (f0 * t + (f1 - f0) * t * t / (2.0 * dur))
        return _np.sin(phase)

    def _noise(self, dur, seed):
        """White noise. Fixed seed -> identical sound every launch
        (and never touches the sim's rng)."""
        rng = _np.random.default_rng(seed)
        return rng.uniform(-1.0, 1.0, int(self.rate * dur))

    def _lowpass(self, x, cutoff):
        """One-pole lowpass. Recursive; startup-only, speed doesn't matter."""
        rc = 1.0 / (2.0 * math.pi * cutoff)
        dt = 1.0 / self.rate
        a = dt / (rc + dt)
        out = _np.empty_like(x)
        y = 0.0
        for i in range(len(x)):
            y += a * (x[i] - y)
            out[i] = y
        return out

    def _loopable(self, x, fade):
        """Crossfade the tail into the head so play(loops=-1) has no click."""
        L = len(x)
        fade = min(fade, L // 2)
        y = _np.empty(L - fade)
        w = _np.arange(fade) / fade
        y[:fade] = x[:fade] * w + x[L - fade:] * (1.0 - w)
        y[fade:] = x[fade:L - fade]
        return y

    def _tone(self, freq, dur, vol=1.0):
        """Sine tone with exponential decay — the no-numpy fallback."""
        n = int(self.rate * dur)
        out = [0.0] * n
        for i in range(n):
            t = i / self.rate
            out[i] = (vol * math.exp(-6.0 * t / dur)
                      * math.sin(2.0 * math.pi * freq * t))
        snd = _to_sound(out, self.rate)
        snd.set_volume(SFX_MASTER_VOLUME)
        return snd

    # --- synthesis: the S2 recipe set ------------------------------------
    def _build_all(self):
        # laser: fast downward sweep, punchy decay
        dur = 0.12
        t = self._t(dur)
        self._reg("laser", self._sweep(900, 300, dur) * _np.exp(-8 * t / dur))

        # beam: hitscan laser discharge — a sharp high zap (brighter and
        # longer than the gun blip; one shot per charge cycle, no throttle)
        dur = 0.18
        t = self._t(dur)
        x = (0.8 * self._sweep(1800, 700, dur)
             + 0.2 * self._lowpass(self._noise(dur, 7), 4000.0))
        self._reg("beam", x * _np.exp(-10 * t / dur))

        # enemy_laser: lower, slower, a bit of grit
        dur = 0.15
        t = self._t(dur)
        x = 0.8 * self._sweep(500, 180, dur) + 0.2 * self._noise(dur, 1)
        self._reg("enemy_laser", x * _np.exp(-6 * t / dur))

        # explosion: lowpassed noise + sub thump
        dur = 0.5
        t = self._t(dur)
        n = self._lowpass(self._noise(dur, 2), 500.0)
        thump = _np.sin(2.0 * math.pi * 55.0 * t)
        self._reg("explosion", (0.8 * n + 0.4 * thump) * _np.exp(-4 * t / dur))

        # small_explosion: brighter, shorter
        dur = 0.25
        t = self._t(dur)
        n = self._lowpass(self._noise(dur, 3), 1200.0)
        self._reg("small_explosion", n * _np.exp(-7 * t / dur))

        # shield_hit: detuned pair = metallic ping, plus an impact tick
        dur = 0.3
        t = self._t(dur)
        ping = (_np.sin(2.0 * math.pi * 1200.0 * t)
                + _np.sin(2.0 * math.pi * 1210.0 * t)) * 0.5
        tick = self._noise(dur, 4) * _np.exp(-40 * t)
        self._reg("shield_hit", ping * _np.exp(-12 * t / dur) + 0.15 * tick)

        # missile_launch: whoosh = lowpassed noise + rising sweep
        dur = 0.4
        t = self._t(dur)
        x = (0.5 * self._lowpass(self._noise(dur, 5), 900.0)
             + 0.5 * self._sweep(200, 600, dur))
        self._reg("missile_launch", x * _np.exp(-3 * t / dur))

        # thruster: 1 s of rumble (filtered noise + 55 Hz), made loopable
        dur = 1.0
        t = self._t(dur)
        n = (0.7 * self._lowpass(self._noise(dur, 6), 300.0)
             + 0.3 * _np.sin(2.0 * math.pi * 55.0 * t))
        self._reg("thruster", self._loopable(n, int(0.05 * self.rate)))

        # game_over: descending stacked-sine sting
        dur = 1.2
        t = self._t(dur)
        ph = 2.0 * math.pi * (400.0 * t - 320.0 * t * t / (2.0 * dur))
        x = (_np.sin(ph) + 0.5 * _np.sin(2 * ph) + 0.25 * _np.sin(3 * ph)) / 1.75
        self._reg("game_over", x * _np.exp(-2.5 * t / dur))

        # placeholder: kept for the TEMP trigger in game.py (removed in S3)
        self._sounds["placeholder"] = self._tone(440.0, 0.15)