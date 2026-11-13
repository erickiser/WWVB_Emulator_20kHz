#!/usr/bin/env python3
"""
wwvb_emulator.py - Generate a WWVB-compatible time signal, disciplined by NTP.

The program outputs a 20 kHz carrier. Its THIRD HARMONIC (3 x 20 kHz = 60 kHz),
produced by distortion in the amplifier/antenna stage, is the WWVB signal.

WHAT IT DOES
    1. Asks an NTP server what time it really is (UTC).
    2. Builds the WWVB "legacy" amplitude-modulated time code for each minute.
    3. Synthesises a 20 kHz carrier with that modulation and plays it out of a
       DAC / sound card (any rate above 40 kHz), or writes a WAV file.

SIGNAL SPEC (NIST legacy AM / pulse-width format)
    * Carrier: 60 000 Hz on air. We generate a 20 000 Hz sine wave and rely on
      its third harmonic. The time-code timing is identical on the harmonic.
      NOTE: a harmonic made by distortion does not shrink in step with the
      fundamental, so the on-air power drop will usually be deeper than 17 dB.
    * One symbol per second. At the START of every UTC second the carrier
      power drops by 17 dB (amplitude x 0.141), then returns to full power after:
            0.2 s -> binary 0
            0.5 s -> binary 1
            0.8 s -> marker (framing)
    * 60 symbols = one frame = one minute. The frame starts at second 0 and
      carries the UTC time of the minute in which it is being sent.
    * Fields are BCD (binary coded decimal), most significant bit first.

    NOT implemented: the phase-modulation (BPSK) time code NIST added in 2012.
    Nearly all consumer "atomic" clocks only decode the AM code above.

REQUIREMENTS
    pip install numpy sounddevice
    (sounddevice is only needed for live playback, not for --wav.)

EXAMPLES
    python wwvb_emulator.py --list-devices
    python wwvb_emulator.py --device 3 --sample-rate 192000
    python wwvb_emulator.py --wav test.wav --duration 120
    python wwvb_emulator.py --wav test.wav --start 2026-03-08T06:59:00 --duration 180
"""

# ---------------------------------------------------------------------------
# Imports. Everything here except numpy ships with Python ("standard library").
# ---------------------------------------------------------------------------
import argparse          # parses command-line options like --device 3
import datetime          # calendar maths (day of year, leap years, ...)
import functools         # gives us lru_cache, a ready-made "remember results" tool
import socket            # raw network access, used to talk to the NTP server
import struct            # converts raw bytes <-> Python numbers
import sys
import threading         # lets the NTP re-sync run in the background
import time
import wave              # writes .wav files

import numpy as np       # fast maths on whole arrays of samples at once

# ---------------------------------------------------------------------------
# Constants. By convention, names in ALL_CAPS are values that never change.
# ---------------------------------------------------------------------------
CARRIER_HZ = 20_000                    # what the DAC outputs (underscores are just for readability)
HARMONIC = 3                           # which harmonic lands on the WWVB frequency
ON_AIR_HZ = CARRIER_HZ * HARMONIC      # 60 000 Hz
LOW_AMPLITUDE = 10 ** (-17 / 20)       # -17 dB of POWER = amplitude ratio of ~0.141

# The three possible symbols. Using names instead of bare numbers makes the
# rest of the code easier to read.
ZERO, ONE, MARKER = 0, 1, 2

# How long the carrier stays at reduced power for each symbol (seconds).
# This is a "dictionary": look up a key (the symbol) to get a value.
LOW_DURATION = {ZERO: 0.2, ONE: 0.5, MARKER: 0.8}

# Seconds within the minute that are always markers.
MARKER_POSITIONS = (0, 9, 19, 29, 39, 49, 59)

# Where each field lives in the frame: (second number, weight of that bit).
# Example: minute 37 = 20 + 10 + 4 + 2 + 1 -> seconds 2, 3, 6, 7, 8 are ONEs.
MINUTE_BITS = ((1, 40), (2, 20), (3, 10), (5, 8), (6, 4), (7, 2), (8, 1))
HOUR_BITS = ((12, 20), (13, 10), (15, 8), (16, 4), (17, 2), (18, 1))
DAY_OF_YEAR_BITS = ((22, 200), (23, 100), (25, 80), (26, 40), (27, 20), (28, 10),
                    (30, 8), (31, 4), (32, 2), (33, 1))
DUT1_BITS = ((40, 8), (41, 4), (42, 2), (43, 1))        # in tenths of a second
YEAR_BITS = ((45, 80), (46, 40), (47, 20), (48, 10),
             (50, 8), (51, 4), (52, 2), (53, 1))        # last two digits of year

NTP_PORT = 123
# NTP counts seconds from 1 Jan 1900; Python/Unix counts from 1 Jan 1970.
NTP_TO_UNIX_SECONDS = 2_208_988_800


# ===========================================================================
# PART 1 - Getting the time from NTP
# ===========================================================================
def query_ntp(server, timeout=2.0):
    """
    Ask one NTP server for the time.

    Returns a "tuple" (two values bundled together):
        offset - seconds to ADD to this computer's clock to get true UTC
        delay  - network round-trip time in seconds (smaller = more trustworthy)

    Raises an exception (Python's way of signalling an error) if it fails.
    """
    # An NTP request is 48 bytes. Only the first byte matters for a simple
    # client: 0x23 means "no leap warning, protocol version 4, I am a client".
    request = bytearray(48)
    request[0] = 0x23

    # "with" guarantees the socket is closed afterwards, even if an error occurs.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:   # DGRAM = UDP
        sock.settimeout(timeout)
        t_send = time.time()                       # our clock when the request leaves
        sock.sendto(request, (server, NTP_PORT))
        reply, _ = sock.recvfrom(512)              # "_" = a value we don't care about
        t_recv = time.time()                       # our clock when the reply arrives

    if len(reply) < 48:
        raise ValueError("NTP reply too short")
    if reply[1] == 0:
        # Stratum 0 is a "kiss of death": the server is refusing to serve us.
        raise ValueError("NTP server refused the request (stratum 0)")

    # Unpack the reply as twelve 32-bit unsigned integers in network byte order
    # ("!" = network order, "12I" = twelve unsigned ints).
    words = struct.unpack("!12I", reply[:48])

    # Each NTP timestamp is two words: whole seconds, then a fraction / 2^32.
    server_recv = words[8] + words[9] / 2**32 - NTP_TO_UNIX_SECONDS
    server_send = words[10] + words[11] / 2**32 - NTP_TO_UNIX_SECONDS

    # Standard NTP formulas. They cancel out the network delay, assuming the
    # trip there takes about as long as the trip back.
    offset = ((server_recv - t_send) + (server_send - t_recv)) / 2
    delay = (t_recv - t_send) - (server_send - server_recv)
    return offset, delay


class NtpClock:
    """
    A clock that returns true UTC, corrected by NTP.

    A "class" bundles data (here: the correction) with the functions that use
    it. Inside a class, "self" means "this particular clock object".

    Design note: we do NOT simply add an offset to time.time(), because the
    operating system may step that clock at any moment. Instead we anchor UTC
    to time.monotonic(), a stopwatch that only ever runs smoothly forward.
    """

    def __init__(self, servers, resync_seconds=600):
        # __init__ runs automatically when you write NtpClock(...).
        self.servers = servers
        self.resync_seconds = resync_seconds
        self._utc_minus_monotonic = None      # filled in by sync()
        self._lock = threading.Lock()         # stops two threads touching the value at once
        self.last_offset = None
        self.last_delay = None

    def sync(self, attempts=4):
        """Query NTP several times and keep the answer with the shortest delay."""
        best = None                            # will hold (delay, utc_minus_monotonic, offset)
        for server in self.servers:
            for _ in range(attempts):
                try:
                    offset, delay = query_ntp(server)
                except (OSError, ValueError):  # network error or bad reply: try again
                    continue
                anchor = time.time() + offset - time.monotonic()
                if best is None or delay < best[0]:
                    best = (delay, anchor, offset)
            if best is not None:
                break                          # this server worked; no need to try others
        if best is None:
            return False                       # nothing answered
        with self._lock:
            self.last_delay, self._utc_minus_monotonic, self.last_offset = best
        return True

    def use_system_clock(self):
        """Fallback: trust the computer's own clock (no NTP correction)."""
        with self._lock:
            self._utc_minus_monotonic = time.time() - time.monotonic()
            self.last_offset, self.last_delay = 0.0, None

    def now(self):
        """Current UTC as Unix time (seconds since 1970, with fractions)."""
        with self._lock:
            return time.monotonic() + self._utc_minus_monotonic

    def start_background_resync(self):
        """Re-query NTP every few minutes so slow clock drift gets corrected."""
        def loop():                            # a function defined inside a function
            while True:
                time.sleep(self.resync_seconds)
                if self.sync():
                    print(f"[ntp] resync: offset {self.last_offset * 1000:+.2f} ms, "
                          f"delay {self.last_delay * 1000:.1f} ms")
                else:
                    print("[ntp] resync failed; continuing on previous correction")
        # daemon=True means this thread won't keep the program alive at exit.
        threading.Thread(target=loop, daemon=True).start()


# ===========================================================================
# PART 2 - Building the 60-symbol time code frame
# ===========================================================================
def nth_sunday(year, month, n):
    """Date of the n-th Sunday of a month (n=1 is the first Sunday)."""
    first = datetime.date(year, month, 1)
    # weekday(): Monday=0 ... Sunday=6. "%" is remainder-after-division.
    days_until_sunday = (6 - first.weekday()) % 7
    return first + datetime.timedelta(days=days_until_sunday + 7 * (n - 1))


def dst_bits(utc_date, mode):
    """
    Return (bit57, bit58) for the frame.

        bit 57 = DST will be in effect at the END of this UTC day
        bit 58 = DST was in effect at the START of this UTC day
    So: 0,0 = standard time   1,1 = DST   1,0 = DST starts today   0,1 = ends today

    'auto' uses current US rules: 2nd Sunday of March to 1st Sunday of November.
    """
    if mode == "on":
        return ONE, ONE
    if mode == "off":
        return ZERO, ZERO
    begins = nth_sunday(utc_date.year, 3, 2)
    ends = nth_sunday(utc_date.year, 11, 1)
    bit57 = ONE if begins <= utc_date < ends else ZERO
    bit58 = ONE if begins < utc_date <= ends else ZERO
    return bit57, bit58


def set_bcd(frame, value, bit_table):
    """
    Write 'value' into the frame using a table of (position, weight) pairs.

    Works from the biggest weight down: if the weight "fits" into what is
    left of the value, that bit is a ONE and we subtract the weight.
    """
    for position, weight in bit_table:
        if value >= weight:
            frame[position] = ONE
            value -= weight


# lru_cache remembers recent results, so asking for the same minute again
# (which happens thousands of times a minute) costs almost nothing.
@functools.lru_cache(maxsize=16)
def build_frame(minute_start, dut1_tenths=0, leap_warning=False, dst_mode="auto"):
    """
    Build the 60 symbols for the minute beginning at Unix time 'minute_start'.
    Returns a tuple of 60 values, each ZERO, ONE or MARKER.
    """
    utc = datetime.datetime.fromtimestamp(minute_start, tz=datetime.timezone.utc)
    day_of_year = utc.timetuple().tm_yday            # 1 Jan = 1
    year = utc.year
    is_leap_year = (year % 4 == 0 and year % 100 != 0) or year % 400 == 0

    frame = [ZERO] * 60                              # a list of sixty ZEROs
    for position in MARKER_POSITIONS:
        frame[position] = MARKER

    set_bcd(frame, utc.minute, MINUTE_BITS)
    set_bcd(frame, utc.hour, HOUR_BITS)
    set_bcd(frame, day_of_year, DAY_OF_YEAR_BITS)
    set_bcd(frame, year % 100, YEAR_BITS)

    # DUT1 = (UT1 - UTC), astronomical correction. Sign is sent as + - + :
    # positive sets seconds 36 and 38, negative sets second 37.
    if dut1_tenths >= 0:
        frame[36] = frame[38] = ONE
    else:
        frame[37] = ONE
    set_bcd(frame, abs(dut1_tenths), DUT1_BITS)

    frame[55] = ONE if is_leap_year else ZERO
    frame[56] = ONE if leap_warning else ZERO        # leap second at end of this month
    frame[57], frame[58] = dst_bits(utc.date(), dst_mode)

    return tuple(frame)        # a tuple is a list that can't be changed (safe to cache)


def frame_to_text(frame):
    """Human-readable frame, e.g. 'M00100101M...' (M = marker)."""
    return "".join("01M"[symbol] for symbol in frame)


# ===========================================================================
# PART 3 - Turning the frame into a waveform
# ===========================================================================
class WwvbSynth:
    """Produces blocks of audio samples for any stretch of UTC time."""

    def __init__(self, sample_rate, level=0.8, dut1_tenths=0,
                 leap_warning=False, dst_mode="auto"):
        self.sample_rate = int(sample_rate)
        self.level = level                 # peak output, 1.0 = DAC full scale
        self.dut1_tenths = dut1_tenths
        self.leap_warning = leap_warning
        self.dst_mode = dst_mode

    def frame_for(self, unix_second):
        """The frame covering a given second."""
        minute_start = unix_second - unix_second % 60
        return build_frame(minute_start, self.dut1_tenths,
                           self.leap_warning, self.dst_mode)

    def generate(self, first_sample_index, utc_of_first_sample, count):
        """
        Make 'count' samples.

        first_sample_index  - running sample counter. The carrier phase comes
                              from this, so the sine wave never glitches even
                              if our idea of UTC is nudged slightly.
        utc_of_first_sample - Unix time at which the first sample is emitted.
                              The on/off envelope comes from this.
        """
        fs = self.sample_rate

        # np.arange(count) is the array [0, 1, 2, ... count-1]. Maths on a
        # numpy array applies to every element at once - no Python loop needed.
        utc = utc_of_first_sample + np.arange(count) / fs
        whole_second = np.floor(utc)
        fraction = utc - whole_second          # 0.0 .. 0.999 position inside the second

        # For each sample, how long is the low-power period of ITS second?
        low_time = np.empty(count)
        for second in np.unique(whole_second):         # usually just 1 or 2 values
            second = int(second)
            symbol = self.frame_for(second)[second % 60]
            # "whole_second == second" is a True/False mask picking those samples.
            low_time[whole_second == second] = LOW_DURATION[symbol]

        # Envelope: low amplitude at the start of the second, full after.
        envelope = np.where(fraction < low_time, LOW_AMPLITUDE, 1.0)

        # Carrier. Taking (index * 20000) modulo the sample rate in whole
        # numbers keeps the phase exact no matter how long the program runs.
        index = first_sample_index + np.arange(count, dtype=np.int64)
        phase = 2 * np.pi * ((index * CARRIER_HZ) % fs) / fs
        return self.level * envelope * np.sin(phase)


# ===========================================================================
# PART 4a - Output to a WAV file (for testing / inspection)
# ===========================================================================
def write_wav(path, synth, start_utc, duration):
    fs = synth.sample_rate
    total = int(duration * fs)
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)          # mono
        wav.setsampwidth(2)          # 2 bytes = 16-bit samples
        wav.setframerate(fs)
        done = 0
        while done < total:
            count = min(fs, total - done)                  # one second at a time
            block = synth.generate(done, start_utc + done / fs, count)
            wav.writeframes((block * 32767).astype("<i2").tobytes())
            done += count
    print(f"Wrote {duration:g} s to {path} ({fs} Hz, 16-bit mono)")


# ===========================================================================
# PART 4b - Live output to the DAC
# ===========================================================================
def play_live(synth, clock, device):
    # Imported here so that --wav mode works without sounddevice installed.
    import sounddevice as sd

    fs = synth.sample_rate
    # "state" holds values the audio callback must remember between calls.
    state = {"samples_sent": 0, "utc_of_sample_zero": None, "fallback_latency": 0.0}

    def callback(outdata, frames, time_info, status):
        """
        The sound system calls this function every few milliseconds, asking
        for the next 'frames' samples. It must be quick and must never wait.
        """
        if status:
            print(f"[audio] {status}", file=sys.stderr)

        # How far in the future will this block actually leave the DAC?
        latency = time_info.outputBufferDacTime - time_info.currentTime
        if not 0 <= latency < 2:                  # some drivers report nonsense
            latency = state["fallback_latency"]
        measured_utc = clock.now() + latency

        # The sound card's crystal and NTP never agree exactly, so we keep a
        # running estimate of "what UTC was sample number 0 played at?" and
        # nudge it gently (2% per block). Big errors are corrected in one jump.
        n = state["samples_sent"]
        if state["utc_of_sample_zero"] is None:
            state["utc_of_sample_zero"] = measured_utc - n / fs
        else:
            error = measured_utc - (state["utc_of_sample_zero"] + n / fs)
            if abs(error) > 0.05:
                state["utc_of_sample_zero"] += error
            else:
                state["utc_of_sample_zero"] += 0.02 * error

        block_utc = state["utc_of_sample_zero"] + n / fs
        outdata[:, 0] = synth.generate(n, block_utc, frames)
        state["samples_sent"] = n + frames

    stream = sd.OutputStream(samplerate=fs, channels=1, dtype="float32",
                             device=device, callback=callback, latency="high")
    state["fallback_latency"] = stream.latency

    with stream:                                   # starts the stream; stops it on exit
        print(f"Outputting {CARRIER_HZ} Hz (harmonic {HARMONIC} = {ON_AIR_HZ} Hz) "
              f"at {fs} samples/s. Ctrl+C to stop.")
        try:
            while True:
                now = clock.now()
                second = int(now)
                stamp = datetime.datetime.fromtimestamp(
                    second - second % 60, tz=datetime.timezone.utc)
                print(f"{stamp:%Y-%m-%d %H:%M} UTC  {frame_to_text(synth.frame_for(second))}")
                time.sleep(60 - now % 60 + 0.05)   # wake just after the next minute starts
        except KeyboardInterrupt:                  # raised when you press Ctrl+C
            print("\nStopped.")


# ===========================================================================
# PART 5 - Command line
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="WWVB time signal emulator disciplined by NTP. Outputs a 20 kHz "
                    "carrier whose third harmonic is the 60 kHz WWVB signal.")
    parser.add_argument("--ntp-server", action="append", metavar="HOST",
                        help="NTP server (repeatable). Default: time.nist.gov, pool.ntp.org")
    parser.add_argument("--resync-minutes", type=float, default=10,
                        help="how often to re-query NTP (default 10)")
    parser.add_argument("--allow-system-clock", action="store_true",
                        help="if NTP is unreachable, use the PC clock instead of quitting")
    parser.add_argument("--sample-rate", type=int, default=192000,
                        help="DAC sample rate in Hz; must exceed 40000 (default 192000)")
    parser.add_argument("--device", help="output device number or name (see --list-devices)")
    parser.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    parser.add_argument("--level", type=float, default=0.8,
                        help="peak output level, 0-1 of full scale (default 0.8)")
    parser.add_argument("--dut1", type=float, default=0.0,
                        help="UT1-UTC correction in seconds, -0.9..0.9 (default 0)")
    parser.add_argument("--leap-second-warning", action="store_true",
                        help="set the 'leap second at end of month' bit")
    parser.add_argument("--dst", choices=("auto", "on", "off"), default="auto",
                        help="DST bits: auto = US rules (default)")
    parser.add_argument("--wav", metavar="FILE", help="write a WAV file instead of playing")
    parser.add_argument("--duration", type=float, default=120,
                        help="seconds of signal for --wav (default 120)")
    parser.add_argument("--start", metavar="YYYY-MM-DDTHH:MM:SS",
                        help="UTC start time for --wav (default: now, from NTP)")
    args = parser.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    # Nyquist: a DAC can only produce frequencies below half its sample rate.
    if args.sample_rate <= 2 * CARRIER_HZ:
        parser.error(f"sample rate must be greater than {2 * CARRIER_HZ} Hz "
                     f"to produce {CARRIER_HZ} Hz")
    if not 0 < args.level <= 1:
        parser.error("--level must be between 0 and 1")
    if not -0.9 <= args.dut1 <= 0.9:
        parser.error("--dut1 must be between -0.9 and 0.9")

    synth = WwvbSynth(args.sample_rate, level=args.level,
                      dut1_tenths=round(args.dut1 * 10),
                      leap_warning=args.leap_second_warning, dst_mode=args.dst)

    # A fixed --start time for a WAV file needs no clock at all.
    if args.wav and args.start:
        start = datetime.datetime.fromisoformat(args.start).replace(
            tzinfo=datetime.timezone.utc).timestamp()
        write_wav(args.wav, synth, start, args.duration)
        return

    servers = args.ntp_server or ["time.nist.gov", "pool.ntp.org"]
    clock = NtpClock(servers, resync_seconds=args.resync_minutes * 60)
    if clock.sync():
        print(f"[ntp] synced: PC clock is off by {clock.last_offset * 1000:+.2f} ms "
              f"(round trip {clock.last_delay * 1000:.1f} ms)")
        clock.start_background_resync()
    elif args.allow_system_clock:
        print("[ntp] WARNING: no NTP server reachable; using the PC clock uncorrected")
        clock.use_system_clock()
    else:
        sys.exit("No NTP server reachable. Check the network, pick another "
                 "--ntp-server, or pass --allow-system-clock.")

    if args.wav:
        write_wav(args.wav, synth, clock.now(), args.duration)
    else:
        device = args.device
        if device is not None and device.isdigit():
            device = int(device)                   # "3" -> device number 3
        play_live(synth, clock, device)


# This line means "only run main() when this file is executed directly",
# not when another Python file imports it to reuse its functions.
if __name__ == "__main__":
    main()
