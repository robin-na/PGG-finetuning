import argparse
import json
from pathlib import Path


def load_game_finished_map(transcripts_path: Path):
    finished_map = {}
    with transcripts_path.open() as f:
        for line in f:
            obj = json.loads(line)
            key = (str(obj.get("experiment")), str(obj.get("participant")))
            finished_map[key] = obj.get("game_finished")
    return finished_map


def extract_content(obj):
    return (
        obj.get("response", {})
        .get("body", {})
        .get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_output",
        default="/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/Persona/misc/batch_persona_type_output_validation.jsonl",
    )
    parser.add_argument(
        "--transcripts",
        default="/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/Persona/transcripts_val.jsonl",
    )
    parser.add_argument(
        "--output",
        default="/Users/robinna/Documents/projects/MultiAgent_LLM/PGG-finetuning/Persona/persona_gpt51_val.jsonl",
    )
    args = parser.parse_args()

    batch_path = Path(args.batch_output)
    transcripts_path = Path(args.transcripts)
    output_path = Path(args.output)

    finished_map = load_game_finished_map(transcripts_path)

    missing_finished = 0
    written = 0
    bad_custom_id = 0

    with batch_path.open() as fin, output_path.open("w") as fout:
        for line in fin:
            obj = json.loads(line)
            custom_id = obj.get("custom_id", "")
            if "__" not in custom_id:
                bad_custom_id += 1
                continue
            game_id, player_id = custom_id.split("__", 1)
            key = (game_id, player_id)
            game_finished = finished_map.get(key)
            if game_finished is None:
                missing_finished += 1
            content = extract_content(obj)
            out = {
                "experiment": game_id,
                "participant": player_id,
                "game_finished": game_finished,
                "text": content,
            }
            fout.write(json.dumps(out))
            fout.write("\n")
            written += 1

    print(f"Written: {written}")
    print(f"Missing game_finished: {missing_finished}")
    print(f"Bad custom_id: {bad_custom_id}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
