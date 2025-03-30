import pandas as pd 
import numpy as np 
import ast
import math

# the directory should be set as the "Files" folder from OSF, before going into "data_and_code".
# https://osf.io/2d56w/files/osfstorage?view_only=d046c1c417024569a8f9fed9e6c8d4d1

df_rounds = pd.read_csv("./data_and_code/data/raw_data/learning_wave/player-rounds.csv")
df_players = pd.read_csv("./data_and_code/data/raw_data/learning_wave/players.csv")
df_analysis_learn = pd.read_csv("./data_and_code/data/processed_data/df_analysis_learn.csv")

def pluralize(quantity, singular="coin", plural="coins"):
    """ Return 'coin' if quantity==1, else 'coins'. """
    return singular if float(quantity) == 1.0 else plural

def coins_to_words(value):
    """Translate the value with the correct form of plurality"""
    value_word = pluralize(value, "coin", "coins")
    if isinstance(value, (int, float)):
        value_str = f"{int(value)} {value_word}" if float(value).is_integer() else f"{value} {value_word}"
        
    else:
    # If the endowment is not numeric, just keep it as is (e.g., "Unknown").
        value_str = f"{value} {value_word}"
    return value_str

def build_avatar_map(players_df):
    """
    Given a players DataFrame with columns: 
      _id, data.avatar
    create a dict: { str(_id): <uppercase-avatar> }
    e.g. if row['_id'] = 101, row['data.avatar'] = 'dog', store '101': 'DOG'
    """
    mapping = {}
    for _, row in players_df.iterrows():
        pid_str = str(row['_id'])
        avatar_name = str(row['data.avatar']).upper()
        mapping[pid_str] = avatar_name
    return mapping

def generate_scenario_transcripts_individual(df):
    """
    Given a DataFrame of public goods game design parameters, returns a dict:
        {
          gameId_1: <transcript describing the design parameters>,
          gameId_2: <transcript describing the design parameters>,
          ...
        }

    Required columns:
      - gameId
      - CONFIG_playerCount
      - CONFIG_numRounds
      - CONFIG_showNRounds
      - CONFIG_endowment
      - CONFIG_multiplier
      - CONFIG_MPCR
      - CONFIG_allOrNothing
      - CONFIG_chat
      - CONFIG_defaultContribProp
      - CONFIG_punishmentExists
      - CONFIG_punishmentCost
      - CONFIG_punishmentMagnitude
      - CONFIG_rewardExists
      - CONFIG_rewardCost
      - CONFIG_rewardMagnitude
      - CONFIG_showOtherSummaries
      - CONFIG_showPunishmentId
      - CONFIG_showRewardId

    We group by gameId, use the first row of each group, then build a textual description.
    """

    transcripts = {}

    for game_id, group in df.groupby('gameId'):
        row = group.iloc[0]

        # Retrieve config columns
        num_players = row.get('CONFIG_playerCount', 'Unknown')
        num_rounds = row.get('CONFIG_numRounds', 'Unknown')
        endowment = row.get('CONFIG_endowment', 'Unknown')
        multiplier = row.get('CONFIG_multiplier', 'Unknown')
        all_or_nothing = row.get('CONFIG_allOrNothing', False)
        chat_enabled = row.get('CONFIG_chat', False)
        default_contrib = row.get('CONFIG_defaultContribProp', False)
        punishment_exists = row.get('CONFIG_punishmentExists', False)
        punishment_cost = row.get('CONFIG_punishmentCost', 0)
        punishment_magnitude = row.get('CONFIG_punishmentMagnitude', 0)
        reward_exists = row.get('CONFIG_rewardExists', False)
        reward_cost = row.get('CONFIG_rewardCost', 0)
        reward_magnitude = row.get('CONFIG_rewardMagnitude', 0)

        # Plural handling for players and rounds
        player_label = "player" if num_players == 1 else "players"

        endowment_str = coins_to_words(value=endowment)
        punishment_cost_str = coins_to_words(value=punishment_cost)
        reward_cost_str = coins_to_words(value=reward_cost)
        punishment_magnitude_str = coins_to_words(value=punishment_magnitude)
        reward_magnitude_str = coins_to_words(value=reward_magnitude)

        lines = []
        # Basic game info
        lines.append(
            f"In this multi-player online public goods game, you will be in a group of {num_players} {player_label}. "
            "You will refer to each other by their avatar (e.g., 'DOG', 'CHICKEN')."
        )


        lines.append(f"Each person is given a set amount of {endowment_str} at the start of each round.")
        lines.append(
            "There will be a public fund that you can choose to contribute to"
            "—you will not be able to see others' contributions before making your own. "
            f"After everyone has contributed, the amount in the public fund will be multiplied by the money multiplier of {multiplier}."
        )
        lines.append(
            "This amount is then evenly divided among the group as the payoff. "
            "You get to keep the payoff in addition to whatever you have left of your private funds."
        )

        # Contribution style
        lines.append("") # blank line

        if default_contrib:
            lines.append(
                f"You start each round with all {endowment_str} in the public fund "
                "and can choose to withdraw these to your private fund. The remaining coins in the public fund will be your contribution."
            )
        else:
            lines.append(
                f"You start each round with all {endowment_str} in your private fund "
                "and can contribute by moving these to the public fund."
            )
        if all_or_nothing:
            lines.append(f"You can choose to either contribute all of your {endowment_str} or nothing.")
        else:
            lines.append(f"You can choose to contribute any integer amount from 0 up to your entire endowment of {endowment_str}.")

        # Chat
        if chat_enabled:
            lines.append("You can communicate with other players.")
        else:
            lines.append("You cannot communicate with other players.")
        
        # Punishment
        lines.append("") # blank line
        lines.append("After contribution and redistribution, you will see how much each player contributed to the public fund. ")
        
        if punishment_exists:
            # Convert punishment cost/magnitude to strings with singular/plural
            lines.append(
                f"After seeing each player's contributions, players can impose deductions on each other. "
                f"Per unit deduction, the punisher spends {punishment_cost_str}, "
                f"causing the punished player to lose {punishment_magnitude_str}." 
            )
        else:
            pass #lines.append("After the contribution and redistribution, you will see how much each player contributed to the public fund.")

        # Reward
        if reward_exists:
            lines.append(
                f"After seeing each player's contributions, players can reward each other. "
                f"Per unit reward, the rewarder spends {reward_cost_str} and grants "
                f"{reward_magnitude_str} of reward to the rewarded player."
            )
        else:
            pass #lines.append("No peer reward mechanism is included in this scenario.")

        lines.append("")
        lines.append("You may exit the game at any time. However, please note that leaving the game before completing the game forfeits any bonuses earned.")

        transcript_text = "\n".join(lines)
        transcripts[game_id] = transcript_text

    return transcripts

def parse_dict(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return {}
    return {}

def format_num(x):
    """ Return integer-like floats as int, else float. """
    if isinstance(x, (int, float)):
        if float(x).is_integer():
            return str(int(x))
        else:
            return str(x)
    return str(x)

def generate_participant_transcript(
    df: pd.DataFrame,
    df_analysis: pd.DataFrame,
    avatar_map: dict,
):
    """
    Participant-View Transcript Generator that returns a DataFrame with columns:
      [gameId, playerId, transcript]

    The transcript for each (game, participant) follows the order:

    1) Focal player's contribution
    2) Redistribution stage
    3) Other players' contribution
    4) Focal player's punishment/reward actions (only how they punish others or reward others)
       - do not mention how much the focal player was punished or rewarded
    5) If show_punish_id or show_reward_id is true, show who punished/rewarded the focal player 
       (and the units)
    6) If show_others is true, for each other player show:
       - their contribution
       - how many units they punished
       - how many coins they lost from penalties
       - how many units they rewarded
       - how many coins they gained from rewards
       - their round payoff

    We assume the following columns in df_analysis for config:
      - CONFIG_multiplier
      - CONFIG_showNRounds
      - CONFIG_showOtherSummaries
      - CONFIG_showPunishmentId
      - CONFIG_showRewardId

    We rely on a function 'generate_scenario_transcripts_individual' that returns a scenario_dict:
      { gameId : scenario_text_for_that_game }

    Usage:
      output_df = generate_participant_transcript_ordered(df, df_analysis, avatar_map)
      # output_df has columns [gameId, playerId, transcript]
    """

    scenario_dict = generate_scenario_transcripts_individual(df_analysis)

    # Final output -> rows of (gameId, playerId, transcript)
    records = []

    for game_id in df["gameId"].unique():
        if game_id not in scenario_dict:
            # No scenario entry found for game ID {game_id}. Skipping transcript.
            continue

        # Attempt to get config row
        config_rows = df_analysis[df_analysis["gameId"] == game_id]
        if len(config_rows) == 0:
            print(f"Warning: No config rows for game {game_id}, skipping.")
            continue
        config_row = config_rows.iloc[0]

        show_nrounds = config_row.get('CONFIG_showNRounds', False)
        multiplier = config_row.get('CONFIG_multiplier', 'Unknown')
        punishment_exists = config_row.get('CONFIG_punishmentExists', False)
        punishment_cost = config_row.get('CONFIG_punishmentCost', 0)
        punishment_magnitude = config_row.get('CONFIG_punishmentMagnitude', 0)
        reward_exists = config_row.get('CONFIG_rewardExists', False)
        reward_cost = config_row.get('CONFIG_rewardCost', 0)
        reward_magnitude = config_row.get('CONFIG_rewardMagnitude', 0)
        show_others = config_row.get('CONFIG_showOtherSummaries', False)
        show_punish_id = config_row.get('CONFIG_showPunishmentId', False)
        show_reward_id = config_row.get('CONFIG_showRewardId', False)

        scenario_text = scenario_dict[game_id]

        # Build for each participant
        game_slice = df[df["gameId"] == game_id].copy()
        player_ids_in_game = game_slice["playerId"].unique()
        round_ids_in_order = game_slice["roundId"].unique()

        for focal_pid in player_ids_in_game:
            lines = [scenario_text, ""]  # Start with scenario text with an empty line
            lines.append("# GAME STARTS")
            lines.append("")

            for round_index, rid in enumerate(round_ids_in_order, start=1):
                if show_nrounds:
                    lines.append(f"## Round {round_index} of {len(round_ids_in_order)}:")
                else:
                    lines.append(f"## Round {round_index}:")

                round_slice = game_slice[game_slice["roundId"] == rid]

                focal_row = round_slice[round_slice["playerId"] == focal_pid]
                others_rows = round_slice[round_slice["playerId"] != focal_pid]

                if len(focal_row) == 1:
                    # (1) Focal player's contribution
                    row_fp = focal_row.iloc[0]
                    cont = row_fp.get("data.contribution", 0.0)
                    payoff_fp = row_fp.get("data.roundPayoff")

                    if math.isnan(payoff_fp):
                        lines.append("You <<exited the game>>.") # exits the game and not mentioned in the next rounds
                        break

                    cont_str = format_num(cont)
                    cont_word = pluralize(cont)

                    lines.append(f"### Contribution Stage: Decide how much to contribute.")
                    lines.append(f"You contributed <<{cont_str} {cont_word}>>.")

                    # (2) Redistribution stage
                    total_contrib = round_slice["data.contribution"].dropna().sum() # dropna before summing
                    public_fund = round(total_contrib * float(multiplier), 1)
                    # e.g. rounding is optional
                    # or int if you prefer
                    public_fund_str = format_num(public_fund)
                    public_fund_word = pluralize(public_fund)

                    lines.append("### Redistribution Stage")
                    lines.append(
                        f"Total group contribution is {format_num(total_contrib)} "
                        f"{pluralize(total_contrib)}, multiplied by {multiplier} => "
                        f"{public_fund_str} {public_fund_word} in the public fund => "
                        f"divided and redistributed to each remaining player."
                    )

                    # (3) List other players' contribution
                    lines.append("### Outcome Stage")
                    for idx_o in others_rows.index:
                        row_o = others_rows.loc[idx_o]
                        pid_o = row_o["playerId"]
                        pid_o_str = str(pid_o)
                        cont_o = row_o.get("data.contribution") # retrieve as nan
                        cont_o_str = format_num(cont_o)
                        cont_o_word = pluralize(cont_o)
                        avatar_o = avatar_map.get(pid_o_str, f"Player {pid_o}")
                        if math.isnan(cont_o):
                            lines.append(f"{avatar_o} is no longer in the game.")
                        else:
                            lines.append(f"{avatar_o} contributed {cont_o_str} {cont_o_word}.")
                    
                    total_punishment_cost = 0 # total punishment cost in this round (will stay 0 if punishment is disabled or not enacted.)
                    total_reward_cost = 0 # total reward cost in this round (will stay 0 if reward is disabled or not enacted.)

                    # (4) Focal player's own punishing/rewarding of others
                    #    do not mention how the focal player was punished/rewarded
                    if punishment_exists and reward_exists:
                        lines.append("### Reward/Deduction Stage: Decide which players, if any, to reward or deduct coins — and how much.")
                        pun_dict = parse_dict(row_fp.get("data.punished", {}))
                        rew_dict = parse_dict(row_fp.get("data.rewarded", {}))

                        if pun_dict or rew_dict:
                            if rew_dict:
                                for rew_pid, unit_rew in rew_dict.items():
                                    if unit_rew > 0: # sometimes the dictionary may be nonempty with 0 unit value
                                        unit_rew_str = format_num(unit_rew)
                                        amt_rew = unit_rew * reward_magnitude
                                        amt_rew_str = format_num(amt_rew)
                                        unit_word = "unit" if float(unit_rew) == 1.0 else "units"
                                        rew_word = "coin" if float(amt_rew) == 1.0 else "coins"
                                        target_name = avatar_map.get(str(rew_pid), f"Player {rew_pid}")
                                        lines.append(f"You <<rewarded {target_name}, granting them {amt_rew_str} {rew_word}>>.")
                                    total_reward_cost += unit_rew * reward_cost  # update total spent on reward
                            if pun_dict:
                                for pun_pid, unit_pun in pun_dict.items():
                                    if unit_pun > 0: # sometimes the dictionary may be nonempty with 0 unit value
                                        unit_pun_str = format_num(unit_pun)
                                        amt_pun = unit_pun * punishment_magnitude
                                        amt_pun_str = format_num(amt_pun)
                                        unit_word = "unit" if float(unit_pun) == 1.0 else "units"
                                        pun_word = "coin" if float(amt_pun) == 1.0 else "coins"
                                        target_name = avatar_map.get(str(pun_pid), f"Player {pun_pid}")
                                        lines.append(f"You <<deducted {amt_pun_str} {pun_word} from {target_name}>>.")#" {target_name} lost {amt_pun_str} {pun_word}.")
                                        total_punishment_cost += unit_pun * punishment_cost  # update total spent on punishment
                        if total_reward_cost + total_punishment_cost <= 0:  # no punishment or reward to anyone
                            lines.append("You <<did not reward or deduct coins from any player>>.")

                    elif punishment_exists:
                        lines.append("### Deduction Stage: Decide which players, if any, to deduct coins — and how much.")
                        pun_dict = parse_dict(row_fp.get("data.punished", {}))
                        if pun_dict:
                            for pun_pid, unit_pun in pun_dict.items():
                                if unit_pun > 0:
                                    unit_pun_str = format_num(unit_pun)
                                    amt_pun = unit_pun * punishment_magnitude
                                    amt_pun_str = format_num(amt_pun)
                                    unit_word = "unit" if float(unit_pun) == 1.0 else "units"
                                    pun_word = "coin" if float(amt_pun) == 1.0 else "coins"
                                    target_name = avatar_map.get(str(pun_pid), f"Player {pun_pid}")
                                    lines.append(f"You <<deducted {amt_pun_str} {pun_word} from {target_name}>>.")#"{target_name} lost {amt_pun_str} {pun_word}.")
                                    total_punishment_cost += unit_pun * punishment_cost  # update total spent on punishment

                        if total_punishment_cost <=0: # if no punishment occured
                            lines.append("You <<did not deduct coins from any player>>.")
                    
                    elif reward_exists:
                        lines.append("### Reward Stage: Decide which players, if any, to reward coins — and how much.")
                        rew_dict = parse_dict(row_fp.get("data.rewarded", {}))
                        if rew_dict:
                            for rew_pid, unit_rew in rew_dict.items():
                                if unit_rew:
                                    unit_rew_str = format_num(unit_rew)
                                    amt_rew = unit_rew * reward_magnitude
                                    amt_rew_str = format_num(amt_rew)
                                    unit_word = "unit" if float(unit_rew) == 1.0 else "units"
                                    rew_word = "coin" if float(amt_rew) == 1.0 else "coins"
                                    target_name = avatar_map.get(str(rew_pid), f"Player {rew_pid}")
                                    lines.append(f"You <<rewarded {target_name}, granting them {amt_rew_str} {rew_word}>>.")
                                    total_reward_cost += unit_rew * reward_cost  # update total spent on reward

                        if total_reward_cost <= 0: # if no reward occured
                            lines.append("You <<did not reward any player>>.")
                    else:
                        pass

                    # (5) If show_punish_id or show_reward_id is true,
                    #     show who punished/rewarded the focal player
                    #     But do not mention how many coins were actually removed/added to the focal player if that's not known yet
                    if punishment_exists and show_punish_id:
                        pun_by_dict = parse_dict(row_fp.get("data.punishedBy", {}))
                        if pun_by_dict:
                            for punisher_id, p_amt in pun_by_dict.items():
                                pun_amt_str = format_num(p_amt * punishment_magnitude)
                                pun_unit = "coin" if float(p_amt* punishment_magnitude) == 1.0 else "coins"
                                pun_name = avatar_map.get(str(punisher_id), f"Player {punisher_id}")
                                if show_punish_id:
                                    lines.append(f"{pun_name} deducted {pun_amt_str} {pun_unit} from you.")

                    if reward_exists and show_reward_id:
                        rew_by_dict = parse_dict(row_fp.get("data.rewardedBy", {}))
                        if rew_by_dict:
                            for rewarder_id, r_amt in rew_by_dict.items():
                                r_amt_str = format_num(r_amt * reward_magnitude)
                                r_unit = "coin" if float(r_amt* reward_magnitude) == 1.0 else "coins"
                                r_name = avatar_map.get(str(rewarder_id), f"Player {rewarder_id}")
                                lines.append(f"{r_name} rewarded you, granting you {r_amt_str} {r_unit}.")

                    # ... existing code for (1) "You contributed X coins" and the redistribution stage ...
                    # ... up to the point where we handle the focal player's summary ...
                    # -------------- FOCAL PLAYER SUMMARY --------------

                    lines.append("### Round Summary")

                    # We'll build one line summarizing the focal player's final stats
                    row_fp = focal_row.iloc[0]

                    # coins deducted from other players -> row_fp["data.penalties"]
                    penalties_inflicted = row_fp.get("data.penalties", 0.0)


                    # coins rewarded to others -> row_fp["data.rewards"]
                    rewards_given = row_fp.get("data.rewards", 0.0)

                    # final payoff
                    payoff_fp = row_fp.get("data.roundPayoff", 0.0)
                    sign_str = "+" if payoff_fp > 0 else ""
                    payoff_str = format_num(payoff_fp)
                    payoff_word = pluralize(payoff_fp)

                    # build them in your requested order, skipping punish/reward lines if zero or mechanism doesn't exist
                    summary_parts = []

                    # a) contributed
                    summary_parts.append(f"You:")


                    # b) coins used for punishment
                    if punishment_exists and total_punishment_cost > 0:
                        pun_used_word = pluralize(total_punishment_cost)
                        summary_parts.append(f"{format_num(total_punishment_cost)} {pun_used_word} spent on deductions.")

                    # c) coins deducted from other players
                    if punishment_exists and penalties_inflicted > 0:
                        pen_infl_word = pluralize(penalties_inflicted)
                        summary_parts.append(f"{format_num(penalties_inflicted)} {pen_infl_word} deducted by other players.")

                    # d) coins used for rewards
                    if reward_exists and total_reward_cost > 0:
                        rew_used_word = pluralize(total_reward_cost)
                        summary_parts.append(f"{format_num(total_reward_cost)} {rew_used_word} spent on rewards.")

                    # e) coins rewarded to others
                    if reward_exists and rewards_given > 0:
                        rew_give_word = pluralize(rewards_given)
                        summary_parts.append(f"{format_num(rewards_given)} {rew_give_word} rewarded other players.")

                    # f) final payoff
                    summary_parts.append(f"Total round gains/losses are {sign_str}{payoff_str} {payoff_word}.")

                    lines.append(" ".join(summary_parts))



                    # -------------- IF SHOW_OTHERS --------------

                    if show_others:

                        for idx_o in others_rows.index:

                            row_o = others_rows.loc[idx_o]

                            payoff_o = row_o.get("data.roundPayoff")

                            pid_o = row_o["playerId"]
                            pid_o_str = str(pid_o)
                            avatar_o = avatar_map.get(pid_o_str, f"Player {pid_o}")

                            if math.isnan(payoff_o):
                                lines.append(f"{avatar_o} is no longer in the game.")
                                continue
                            else:
                                pass

                            # 1) contribution

                            # 2) pun_dict for how many units they inflicted
                            pun_dict_o = parse_dict(row_o.get("data.punished", {}))
                            pun_units_o = sum(pun_dict_o.values())
                            pun_coins_used_o = pun_units_o * punishment_cost
                            pun_coins_inflicted_o = row_o.get("data.penalties", 0.0)

                            # 3) rew_dict for how many units they granted
                            rew_dict_o = parse_dict(row_o.get("data.rewarded", {}))
                            rew_units_o = sum(rew_dict_o.values())
                            rew_coins_used_o = rew_units_o * reward_cost
                            rew_coins_given_o = row_o.get("data.rewards", 0.0)

                            # 4) final payoff
                            sign_str = "+" if payoff_o > 0 else ""
                            payoff_str = format_num(payoff_o)
                            payoff_word_ = pluralize(payoff_o)

                            summary_parts = [
                                f"{avatar_o}:"
                            ]

                            if punishment_exists and pun_units_o > 0:
                                pun_used_word_o = pluralize(pun_coins_used_o)
                                summary_parts.append(f"{format_num(pun_coins_used_o)} {pun_used_word_o} spent on deductions.")
                            
                            if punishment_exists and pun_coins_inflicted_o > 0:
                                pun_infl_word_o = pluralize(pun_coins_inflicted_o)
                                summary_parts.append(f"{format_num(pun_coins_inflicted_o)} {pun_infl_word_o} deducted by other players.")

                            if reward_exists and rew_units_o > 0:
                                rew_used_word_o = pluralize(rew_coins_used_o)
                                summary_parts.append(f"{format_num(rew_coins_used_o)} {rew_used_word_o} spent on rewards.")

                            if reward_exists and rew_coins_given_o > 0:
                                rew_give_word_o = pluralize(rew_coins_given_o)
                                summary_parts.append(f"{format_num(rew_coins_given_o)} {rew_give_word_o} rewarded by other players.")

                            summary_parts.append(f"Total round gains/losses are {sign_str}{payoff_str} {payoff_word_}.")
                            lines.append(" ".join(summary_parts))

                lines.append("")  # End of round

            lines.append("# GAME COMPLETE")
            # Build final transcript
            transcript_text = "\n".join(lines)
            records.append((game_id, focal_pid, transcript_text))

    output_df = pd.DataFrame(records, columns=["experiment","participant","text"])
    output_df.to_json('prompts.jsonl', orient='records', lines=True)

    return None

## generate the prompts.jsonl file
avatar_map = build_avatar_map(df_players)
generate_participant_transcript(df=df_rounds, df_analysis=df_analysis_learn, avatar_map=avatar_map)

