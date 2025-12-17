#!/usr/bin/env python3
"""
Exhaustive frame loading test - loads every single frame in the dataset.
Only outputs failures to keep logs clean.
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from tqdm import tqdm

    sys.path.insert(0, "src")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print("Loading dataset...")
    dataset = LeRobotDataset(
        "eliasab16/xvlm_insert_tip_into_mounted_device",
        root=Path.home() / ".cache/huggingface/lerobot",
        video_backend="torchcodec",
    )

    total_frames = len(dataset)
    print(f"Dataset loaded: {total_frames:,} frames")
    print(f"Testing all frames (this may take a while)...\n")

    failed_frames = []
    
    # Test every single frame
    for idx in tqdm(range(total_frames), desc="Testing frames", unit="frame"):
        try:
            item = dataset[idx]
            # Successfully loaded - don't print anything
        except Exception as e:
            # Failed - record it
            try:
                ep_idx = dataset.hf_dataset[idx]["episode_index"].item()
                ts = dataset.hf_dataset[idx]["timestamp"].item()
            except:
                ep_idx = "unknown"
                ts = "unknown"
            
            failed_frames.append({
                "frame_index": idx,
                "episode": ep_idx,
                "timestamp": ts,
                "error": str(e),
                "error_type": type(e).__name__,
            })
            
            # Print error immediately
            tqdm.write(f"\n✗ FAILED: Frame {idx} (episode {ep_idx}, timestamp {ts})")
            tqdm.write(f"  Error: {type(e).__name__}: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Total frames tested: {total_frames:,}")
    print(f"Failed frames: {len(failed_frames)}")
    
    if len(failed_frames) == 0:
        print("\n✓ SUCCESS: All frames loaded successfully!")
        print("\nConclusion: Your dataset is NOT corrupted.")
        print("The training error is likely due to:")
        print("  1. Shuffling causing different frame order in training")
        print("  2. Environment differences (CUDA vs CPU, different libraries)")
        print("  3. Torchcodec multi-worker race condition")
        print("\nRecommended solutions:")
        print("  • Try: --dataset.video_backend=pyav")
        print("  • Try: --num_workers=0")
    else:
        print("\n✗ FAILURE: Some frames could not be loaded")
        print("\nFailed frames details:")
        for f in failed_frames:
            print(f"\n  Frame {f['frame_index']}:")
            print(f"    Episode: {f['episode']}")
            print(f"    Timestamp: {f['timestamp']}")
            print(f"    Error: {f['error_type']}: {f['error']}")
        
        print("\nRecommended action:")
        print("  • Re-record the episodes with corrupted frames")
        print("  • Or exclude these episodes from training")
    
    print("=" * 80)
