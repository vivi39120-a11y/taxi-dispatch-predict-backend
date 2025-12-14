from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parent

def run(script_name: str):
    script = BASE / script_name
    if not script.exists():
        raise FileNotFoundError(script)
    print(f"\n=== RUN: {script_name} ===")
    subprocess.check_call([sys.executable, str(script)])

def main():
    run("predict_next_hour_advanced.py")
    run("rank_next_hour.py")
    run("build_driver_view_map_interactive.py")

    public_dir = BASE / "outputs" / "public"
    print("\n✅ Pipeline finished.")
    print("✅ Frontend 可直接連結的檔案位置：")
    print(" -", public_dir / "index.html")
    print(" -", public_dir / "driver_view_simple.html")
    print(" -", public_dir / "zones.json")
    print(" -", public_dir / "meta.json")

if __name__ == "__main__":
    main()
