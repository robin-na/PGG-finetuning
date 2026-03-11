# Post-hoc cluster profile

This report characterizes the current six clusters using learning-wave actual player-round behavior plus the tagged LLM archetype summaries. Behavioral averages are soft-cluster weighted, not hard-cluster averages.

## cluster_1 / `strong_mildly_conditional_cooperator`

- mass: mean `0.226`, top-assignment share `0.226`
- contribution: mean rate `0.774`, zero rate `0.106`, full rate `0.634`, volatility `0.193`, endgame delta `+0.008`
- actions given: punish row rate `0.244`, reward row rate `0.076`, punish units/round `0.988`, reward units/round `0.358`
- actions received: punished row rate `0.209`, rewarded row rate `0.160`, payoff/endowment `3.967`

Tag profile:
- communication: presence `0.061`, avg chars `284.2`, keywords `style, unknown, communication, chat, messages, engagement, behavior, attempt`
  snippet: Unknown.
- punishment: presence `1.000`, avg chars `1380.6`, keywords `punishment, punish, norm, low, contributors, player, use, free`
  snippet: The player is an active, norm-enforcing punisher who reliably spends to sanction low contributors.
- reward: presence `0.529`, avg chars `1073.9`, keywords `reward, rewards, high, use, behavior, positive, rewarding, cooperation`
  snippet: This player uses rewards rarely and selectively compared to punishment.
- response_to_end_game: presence `0.490`, avg chars `892.1`
  snippet: This player shows no end-game decay in cooperation.

## cluster_2 / `opportunistic_free_rider`

- mass: mean `0.132`, top-assignment share `0.132`
- contribution: mean rate `0.338`, zero rate `0.488`, full rate `0.221`, volatility `0.256`, endgame delta `-0.021`
- actions given: punish row rate `0.055`, reward row rate `0.048`, punish units/round `0.148`, reward units/round `0.244`
- actions received: punished row rate `0.199`, rewarded row rate `0.108`, payoff/endowment `3.656`

Tag profile:
- communication: presence `0.702`, avg chars `973.9`, keywords `social, low, chat, group, communication, style, non, player`
  snippet: They are almost completely silent during play and only speak once, at the very end, with a brief, neutral, system-like statement about the impending conclusion of the game.
- punishment: presence `0.410`, avg chars `1079.5`, keywords `punishment, low, use, norm, punish, non, free, contributors`
  snippet: This player is willing to spend on punishment and uses it multiple times, but their pattern is inconsistent with standard norm-enforcing cooperation.
- reward: presence `0.405`, avg chars `957.4`, keywords `reward, rewards, use, rewarding, behavior, high, cooperation, cost`
  snippet: They do not reward anyone, despite the presence of multiple high contributors and visible reward activity by others.
- response_to_end_game: presence `0.635`, avg chars `904.6`
  snippet: The player shows no clear end-game decay in cooperation; instead, their final-round behavior becomes more cooperative than their earlier choices.

## cluster_3 / `fragile_high_cooperator`

- mass: mean `0.179`, top-assignment share `0.180`
- contribution: mean rate `0.867`, zero rate `0.082`, full rate `0.789`, volatility `0.215`, endgame delta `+0.092`
- actions given: punish row rate `0.065`, reward row rate `0.062`, punish units/round `0.254`, reward units/round `0.425`
- actions received: punished row rate `0.049`, rewarded row rate `0.138`, payoff/endowment `4.621`

Tag profile:
- communication: presence `0.996`, avg chars `1285.1`, keywords `group, cooperative, social, chat, low, style, coordination, tone`
  snippet: The player is verbally pro-cooperation and norm-reinforcing.
- punishment: presence `0.433`, avg chars `1241.1`, keywords `punishment, norm, punish, use, group, low, free, non`
  snippet: The player is willing to incur repeated personal costs to punish norm violations and disruptive behavior, but they apply punishment selectively rather than indiscriminately.
- reward: presence `0.402`, avg chars `1130.8`, keywords `reward, rewards, rewarding, use, cooperation, behavior, high, cooperative`
  snippet: Reward is their main “signature.” They spend a very large amount of coins over the game on rewards, far more than most players, and they accept substantial personal losses to do it—even to the point of negative or nea...
- response_to_end_game: presence `0.376`, avg chars `955.1`
  snippet: The player clearly tracks the game horizon: they comment on remaining rounds and respond to others mentioning how many are left.

## cluster_4 / `near_unconditional_full_cooperator`

- mass: mean `0.205`, top-assignment share `0.205`
- contribution: mean rate `0.982`, zero rate `0.007`, full rate `0.963`, volatility `0.044`, endgame delta `+0.020`
- actions given: punish row rate `0.086`, reward row rate `0.080`, punish units/round `0.298`, reward units/round `0.614`
- actions received: punished row rate `0.044`, rewarded row rate `0.166`, payoff/endowment `4.641`

Tag profile:
- communication: presence `0.994`, avg chars `929.8`, keywords `group, cooperative, low, chat, social, norm, style, tone`
  snippet: The player’s communication style is minimal, reactive, and polite.
- punishment: presence `0.580`, avg chars `1028.0`, keywords `punishment, norm, punish, use, free, group, non, low`
  snippet: The player chooses not to use costly punishment, despite clear evidence that some others contribute very little or nothing.
- reward: presence `0.548`, avg chars `950.9`, keywords `reward, rewards, use, rewarding, cooperation, high, behavior, cooperative`
  snippet: They never spend resources to reward others, even in a fully cooperative group where rewarding could signal gratitude or reinforce behavior.
- response_to_end_game: presence `0.562`, avg chars `790.6`
  snippet: The player shows no end-game decline in cooperation.

## cluster_5 / `unknown_or_sparse_info`

- mass: mean `0.035`, top-assignment share `0.035`
- contribution: mean rate `0.480`, zero rate `0.081`, full rate `0.103`, volatility `0.154`, endgame delta `+0.131`
- actions given: punish row rate `0.012`, reward row rate `0.012`, punish units/round `0.145`, reward units/round `0.078`
- actions received: punished row rate `0.031`, rewarded row rate `0.021`, payoff/endowment `3.864`

Tag profile:
- communication: presence `0.467`, avg chars `311.1`, keywords `unknown, sociability, communication, use communication, communication strategically, attempts coordinate, attempts, coordinate contributions`
  snippet: Unknown.
- punishment: presence `0.533`, avg chars `337.9`, keywords `punishment, unknown, player, cost, low, tolerance, use, targeting`
  snippet: Unknown.
- reward: presence `0.504`, avg chars `313.3`, keywords `reward, unknown, rewarding, cost, gratitude, behavior, rewards, high`
  snippet: Unknown.
- response_to_end_game: presence `0.504`, avg chars `310.5`
  snippet: Unknown.

## cluster_6 / `moderate_payoff_aware_conditional_cooperator`

- mass: mean `0.222`, top-assignment share `0.222`
- contribution: mean rate `0.678`, zero rate `0.190`, full rate `0.566`, volatility `0.243`, endgame delta `-0.026`
- actions given: punish row rate `0.000`, reward row rate `0.158`, punish units/round `0.000`, reward units/round `0.615`
- actions received: punished row rate `0.000`, rewarded row rate `0.269`, payoff/endowment `3.869`

Tag profile:
- communication: presence `0.000`, avg chars `322.0`, keywords `cooperative, express, style, low, tone, moralizing, communication, communication style`
  snippet: Unknown.
- punishment: presence `0.000`, avg chars `1026.3`, keywords `punishment, non, norm, clear, respond, avoiding, use, free`
  snippet: They never use the punishment option, despite experiencing instances where some others contribute nothing and thereby benefit from the contributor’s generosity.
- reward: presence `0.527`, avg chars `1424.4`, keywords `reward, rewards, rewarding, use, behavior, high, cooperation, cooperative`
  snippet: This player is a **light but purposeful rewarder** who selectively uses the reward mechanism and does not treat it as a core strategic tool.
- response_to_end_game: presence `0.517`, avg chars `1072.4`
  snippet: The player shows no end-game decay; if anything, they strengthen cooperation as the game horizon approaches.
