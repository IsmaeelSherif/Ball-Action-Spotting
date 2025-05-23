import argparse
from pathlib import Path

from tqdm import tqdm
import numpy as np


import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
print('added to sys.path')

from src.ball_action.annotations import raw_predictions_to_actions, prepare_game_spotting_results
from src.utils import get_best_model_path, get_video_info
from src.predictors import MultiDimStackerPredictor
from src.frame_fetchers import NvDecFrameFetcher
from src.ball_action import constants


RESOLUTION = "720p"
INDEX_SAVE_ZONE = 1
TTA = True


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True, type=str)
    parser.add_argument("--video_path", required=True, type=str)
    parser.add_argument("--weights_dir", required=True, type=str)
    parser.add_argument("--folds", default="all", type=str)
    return parser.parse_args()


def get_raw_predictions(predictor: MultiDimStackerPredictor,
                        video_path: Path,
                        frame_count: int) -> tuple[list[int], np.ndarray]:
    frame_fetcher = NvDecFrameFetcher(video_path, gpu_id=predictor.device.index)
    frame_fetcher.num_frames = frame_count

    indexes_generator = predictor.indexes_generator
    min_frame_index = indexes_generator.clip_index(0, frame_count, INDEX_SAVE_ZONE)
    max_frame_index = indexes_generator.clip_index(frame_count, frame_count, INDEX_SAVE_ZONE)
    frame_index2prediction = dict()
    predictor.reset_buffers()
    with tqdm() as t:
        while True:
            frame = frame_fetcher.fetch_frame()
            frame_index = frame_fetcher.current_index
            prediction, predict_index = predictor.predict(frame, frame_index)
            if predict_index < min_frame_index:
                continue
            if prediction is not None:
                frame_index2prediction[predict_index] = prediction.cpu().numpy()
            t.update()
            if predict_index == max_frame_index:
                break
    predictor.reset_buffers()
    frame_indexes = sorted(frame_index2prediction.keys())
    raw_predictions = np.stack([frame_index2prediction[i] for i in frame_indexes], axis=0)
    return frame_indexes, raw_predictions


def predict_video(predictor: MultiDimStackerPredictor,
                  half: int,
                  video_path: Path,
                  game_prediction_dir: Path,
                  use_saved_predictions: bool) -> dict[str, tuple]:
    video_info = get_video_info(video_path)
    print("Video info:", video_info)
    # assert video_info["fps"] == constants.video_fps

    raw_predictions_path = game_prediction_dir / f"{half}_raw_predictions.npz"

    if use_saved_predictions:
        with np.load(str(raw_predictions_path)) as raw_predictions:
            frame_indexes = raw_predictions["frame_indexes"]
            raw_predictions = raw_predictions["raw_predictions"]
    else:
        print("Predict video:", video_path)
        frame_indexes, raw_predictions = get_raw_predictions(
            predictor, video_path, video_info["frame_count"]
        )
        np.savez(
            raw_predictions_path,
            frame_indexes=frame_indexes,
            raw_predictions=raw_predictions,
        )
        print("Raw predictions saved to", raw_predictions_path)

    class2actions = raw_predictions_to_actions(frame_indexes, raw_predictions)
    return class2actions


def predict_game(predictor: MultiDimStackerPredictor,
                 video_path: str,
                 prediction_dir: Path,
                 use_saved_predictions: bool):
    game = "game"
    game_prediction_dir = prediction_dir / game
    game_prediction_dir.mkdir(parents=True, exist_ok=True)
    print("Predict game:", video_path)

    half2class_actions = dict()
    half2class_actions[2] = dict()
    
    class_actions = predict_video(
        predictor, 1, video_path, game_prediction_dir, use_saved_predictions
    )
    half2class_actions[1] = class_actions

    video_info = get_video_info(video_path)
    print('using fps', video_info['fps'])
    prepare_game_spotting_results(half2class_actions, game, prediction_dir, video_info['fps'])


def predict_fold(experiment: str, fold: int, gpu_id: int,
                 challenge: bool, use_saved_predictions: bool, pred_dir, video_path, weights_dir):
    print(f"Predict games: {experiment=}, {fold=}, {gpu_id=} {challenge=}")
    weights_dir = Path(weights_dir)
    experiment_dir = weights_dir / f"fold_{fold}"
    print('experiment_dir', experiment_dir)

    p = Path(experiment_dir)
    for item in p.iterdir():
        print('item', item)

    model_path = get_best_model_path(experiment_dir)
    print("Model path:", model_path)
    predictor = MultiDimStackerPredictor(model_path, device=f"cuda:{gpu_id}", tta=TTA)
    
    pred_dir = Path(pred_dir)
    prediction_dir = pred_dir / f"fold_{fold}"
    if not prediction_dir.exists():
        prediction_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Folder {prediction_dir} already exists.")

    predict_game(predictor, video_path, prediction_dir, use_saved_predictions)


if __name__ == "__main__":
    args = parse_arguments()

    if args.folds == "all":
        folds = constants.folds
    else:
        folds = [int(fold) for fold in args.folds.split(",")]

    experiment = "ball_finetune_long_004"
    challenge = True
    use_saved_predictions = False
    gpu_id = 0

    for fold in folds:
        predict_fold(experiment, fold, gpu_id, challenge, use_saved_predictions, args.pred_dir, args.video_path, args.weights_dir)
