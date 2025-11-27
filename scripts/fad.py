import argparse

from frechet_audio_distance import FrechetAudioDistance

parser = argparse.ArgumentParser()
parser.add_argument("--bg_dir", type=str, required=True, help="Path to background audio directory")
parser.add_argument("--eval_dir", type=str, required=True, help="Path to evaluation audio directory")
args = parser.parse_args()

frechet = FrechetAudioDistance(
    ckpt_dir="../checkpoints/clap",
    model_name="clap",
    submodel_name="630k-audioset",  # for CLAP only
    sample_rate=48000,
    # use_pca=False, # for VGGish only
    # use_activation=False, # for VGGish only
    verbose=False,
    audio_load_worker=8,
    enable_fusion=False,  # for CLAP only
)

fad_score = frechet.score(args.bg_dir, args.eval_dir)
print("FAD score: %.8f" % fad_score)
