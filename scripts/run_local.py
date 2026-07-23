import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.inference import predict_emotion


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/run_local.py <path_to_audio.wav>")
        sys.exit(1)

    audio_path = sys.argv[1]
    predictions = predict_emotion(audio_path)

    print(f"\nPredictions for: {audio_path}\n")
    for pred in predictions:
        print(f"  {pred['label']:<12} {pred['score'] * 100:.2f}%")


if __name__ == "__main__":
    main()