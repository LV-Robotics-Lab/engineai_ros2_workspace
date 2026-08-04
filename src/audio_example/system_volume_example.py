import argparse
import re
import subprocess
import sys

def get_system_volume() -> int | None:
    """Get the current system default volume percentage via pactl."""
    try:
        # Query the default output device (Sink) status with pactl
        result = subprocess.run(
            ["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
            capture_output=True,
            text=True,
            check=True,
        )

        # Example pactl output: "Volume: front-left: 32768 /  50% / -18.06 dB, ..."
        # Extract the first percentage with a regex
        match = re.search(r"(\d+)%", result.stdout)
        if match:
            volume = int(match.group(1))
            return volume
        else:
            print("⚠️ Unable to parse system volume output format")
            return None

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to get volume: {e}")
        return None
    except FileNotFoundError:
        print("❌ pactl not found; please check PulseAudio / PipeWire tools")
        return None

def set_system_volume(volume: int) -> None:
    """Set the system master volume (0 - 100) via pactl."""
    try:
        subprocess.run(
            ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{volume}%"],
            check=True,
        )
        print(f"🔊 System master volume set to: {volume}%")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to set volume: {e}")
    except FileNotFoundError:
        print("❌ pactl not found; please check PulseAudio / PipeWire tools")

def main():
    parser = argparse.ArgumentParser(
        description="Ubuntu system master volume control (get and set)"
    )

    # 1. Positional argument for setting volume
    parser.add_argument(
        "volume",
        nargs="?",  # '?' means the argument is optional
        type=int,
        help="Target volume (integer 0-100). If omitted and -g is not set, prints current volume",
    )

    # 2. Flag to query volume
    parser.add_argument(
        "-g",
        "--get",
        action="store_true",
        help="Only get the current system master volume",
    )

    args = parser.parse_args()

    # Prefer --get, or query by default when volume is not provided
    if args.get or args.volume is None:
        current_vol = get_system_volume()
        if current_vol is not None:
            print(f"🔈 Current system master volume: {current_vol}%")
        return

    # Validate range and set volume when volume is provided
    if not (0 <= args.volume <= 100):
        print(f"⚠️ Error: volume must be between 0 and 100! Got: {args.volume}")
        sys.exit(1)

    set_system_volume(args.volume)

if __name__ == "__main__":
    main()
