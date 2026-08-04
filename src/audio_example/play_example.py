import argparse

from pygame import mixer

def main():
    parser = argparse.ArgumentParser(description="Play a local audio file")
    parser.add_argument("audio_file", help="Path to the audio file to play")
    args = parser.parse_args()
    audio_file = args.audio_file

    # 1. Initialize the mixer module
    mixer.init()

    try:
        # 2. Load the audio file
        mixer.music.load(audio_file)
        print(f"Successfully loaded audio file: {audio_file}")
    except Exception as e:
        print(f"Failed to load audio, please check the file path: {e}")
        return

    # Default volume (range 0.0 to 1.0)
    current_volume = 0.5
    mixer.music.set_volume(current_volume)

    # 3. Start playback
    mixer.music.play()
    print("\n--- Audio playback controls ---")
    print(" [p] Pause")
    print(" [r] Resume")
    print(" [+] Volume up (+10%)")
    print(" [-] Volume down (-10%)")
    print(" [s] Stop")
    print(" [q] Quit")

    # Interactive control loop
    while True:
        cmd = input("\nEnter command: ").strip().lower()

        if cmd == "p":
            mixer.music.pause()
            print("⏸️  Paused")

        elif cmd == "r":
            mixer.music.unpause()
            print("▶️  Resumed")

        elif cmd == "+":
            current_volume = min(1.0, current_volume + 0.1)
            mixer.music.set_volume(current_volume)
            print(f"🔊 Volume up: {int(current_volume * 100)}%")

        elif cmd == "-":
            current_volume = max(0.0, current_volume - 0.1)
            mixer.music.set_volume(current_volume)
            print(f"🔉 Volume down: {int(current_volume * 100)}%")

        elif cmd == "s":
            mixer.music.stop()
            print("⏹️  Stopped")

        elif cmd == "q":
            mixer.music.stop()
            mixer.quit()
            print("👋 Exited")
            break

        else:
            print("⚠️ Unknown command, please enter p, r, +, -, s, or q")

if __name__ == "__main__":
    main()
