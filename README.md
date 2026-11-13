# WWVB_Emulator_20kHz

Generates a WWVB-compatible time signal, disciplined by NTP, for output through
a DAC driving a ferrite loop antenna. The program outputs a 20 kHz carrier; its
third harmonic (60 kHz), produced by distortion in the output stage, is the WWVB
signal. Any sample rate above 40 kHz works (default 192 kHz).

Implements the legacy NIST amplitude-modulated (pulse-width) time code:
carrier power drops 17 dB at the start of each second and returns after
0.2 s (0), 0.5 s (1) or 0.8 s (marker). The phase-modulated code is not implemented.

## Usage

    pip install -r requirements.txt
    python wwvb_emulator.py --list-devices
    python wwvb_emulator.py --device 3 --sample-rate 192000
    python wwvb_emulator.py --wav test.wav --duration 120

Run `python wwvb_emulator.py --help` for all options.

## Note

Keep the radiated field tiny (room range). Unlicensed emissions at 60 kHz are
subject to FCC Part 15 limits.
