# Trajectory Completion Prompt Layout

## Picture

```text
+----------------------------------------------------------------------------------+
| SYSTEM MESSAGE                                                                   |
| "You forecast the remaining rounds of an observed public goods game ...          |
|  Return only JSON that matches the provided schema."                             |
+----------------------------------------------------------------------------------+
| USER MESSAGE                                                                     |
|                                                                                  |
| GAME SNAPSHOT                                                                    |
| - game_id                                                                        |
| - total_rounds                                                                   |
| - observed_rounds = k                                                            |
| - remaining_rounds                                                               |
| - players                                                                        |
| - endowment                                                                      |
| - all_or_nothing                                                                 |
| - punishment_enabled                                                             |
| - reward_enabled                                                                 |
|                                                                                  |
| TASK                                                                             |
| - predict rounds k+1 ... T                                                       |
| - use only the observer transcript prefix                                        |
| - do not predict chat                                                            |
|                                                                                  |
| STRICT OUTPUT RULES                                                              |
| - JSON only                                                                      |
| - exact game_id                                                                  |
| - exact observed_rounds                                                          |
| - exact number of future rounds                                                  |
| - every round must include every player                                          |
| - contribution is integer 0..20 (or {0,20} if all-or-nothing)                   |
| - punish/reward are arrays of {target, units}                                    |
| - empty array when no action / mechanism disabled                                |
|                                                                                  |
| OBSERVED TRANSCRIPT PREFIX                                                       |
| <original observer transcript header>                                            |
| # OBSERVED PREFIX ENDS HERE                                                      |
| <ROUND i="1 of T"> ... </ROUND>                                                  |
| ...                                                                              |
| <ROUND i="k of T"> ... </ROUND>                                                  |
| # END OBSERVED PREFIX                                                            |
+----------------------------------------------------------------------------------+
| STRUCTURED OUTPUT ENFORCEMENT                                                    |
| OpenAI Batch request uses /v1/chat/completions + response_format=json_schema     |
| so the model must emit a parseable JSON object.                                  |
+----------------------------------------------------------------------------------+
```

## Returned JSON Shape

```json
{
  "game_id": "GAME_ID",
  "observed_rounds": 3,
  "predicted_rounds": [
    {
      "round_number": 4,
      "player_predictions": [
        {
          "player": "SLOTH",
          "contribution": 20,
          "punish": [{"target": "OWL", "units": 1}],
          "reward": []
        }
      ]
    }
  ]
}
```

## Notes

- `punish` and `reward` are always present for every player.
- If a mechanism is disabled, its array is constrained to be empty by the JSON schema.
- Chat stays in the observed prefix when present, but the model is not asked to predict future chat.
