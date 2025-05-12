# Atari Mario Bros – RAM Byte Map


## RAM Layout (Approximate)

| RAM Address | Field Name             | Description                     | Value         | Notes / Conditions                |
| ----------- | ---------------------- | ------------------------------- | ------------- | --------------------------------- |
| 7           | `lives`                | Number of remaining lives       | 5             | HUD only                          |
| 9, 10       | `score_digits`         | BCD-encoded score               | e.g. 2, 4     |                                    |
| 27          | `jumping_state`        | jump state                      | 0 - 150       |                                    |
| 42, 44      | `player_y`, `player_x` | Player position                 | –             | Updated constantly                |
| 101–108     | `bonus_coin_flags`     | Coin active (1) or not (0)      | 0/1           |                          |
| 107–110     | `pest_states`          | State/animation values         | –             | Determines enemy type and frame   |
| 111         | `bonus_block_flag`     | Shows if bonus block is active  | 0/1           | –                                 |
| 115         | `fireball_x`           | Fireball X coordinate           | Range of width| present `117` for full position |
| 117         | `fireball_lane`        | Fireball Y layer (0–3)          | –             | Multiplied by 40 then offset      |
| 121         | `pow_block_flag`       | 0 = gone, >0 = present          | 0–3?          | –                                 |







## Other Constants

| Description       | Position (x, y) | Size (w × h) | Value | Notes                            |
| ----------------- | --------------- | ------------ | ----- | -------------------------------- |
| **Player**        | (37, 100)       | (9 × 21)     | –     | initial spawn position           |
| **Fireball**      | (154, 84)       | (9 × 14)     | –     | spawns later                    |
| **Pow Block**     | (72, 141)       | (16 × 7)     | –     | present if `RAM[121] > 0` |
| **Platform 1**    | (0, 57)         | (64 × 3)     | –     | Bottom layer                     |
| **Platform 2**    | (96, 57)        | (68 × 3)     | –     | Bottom layer                     |
| **Platform 3**    | (31, 95)        | (97 × 3)     | –     | Middle platform                  |
| **Platform 4**    | (0, 95)         | (16 × 3)     | –     | Side connector                   |
| **Platform 5**    | (144, 95)       | (18 × 3)     | –     | Side connector                   |
| **Platform 6**    | (0, 135)        | (48 × 3)     | –     | Top left                         |
| **Platform 7**    | (112, 135)      | (48 × 3)     | –     | Top right                        |
| **Bonus Coin 1**  | (136, 26)       | (9 × 13)     | –     | Bonus  only                 |
| **Bonus Coin 2**  | (151, 68)       | (9 × 13)     | –     | Bonus  only                 |
| **Bonus Coin 3**  | (98, 104)       | (9 × 13)     | –     | Bonus  only                 |
| **Bonus Coin 4**  | (148, 148)      | (9 × 13)     | –     | Bonus  only                 |
| **Bonus Coin 5**  | (16, 26)        | (9 × 13)     | –     | Bonus  only                 |
| **Bonus Coin 6**  | (1, 68)         | (9 × 13)     | –     | Bonus  only                 |
| **Bonus Coin 7**  | (54, 104)       | (9 × 13)     | –     | Bonus  only                 |
| **Bonus Coin 8**  | (4, 148)        | (9 × 13)     | –     | Bonus  only                 |
| **Score Display** | (55, 12)        | (48 × 9)     | 0     |                                |
| **Lives Display** | (71, 12)        | (32 × 9)     | 5     |                              |
| **Timer**         | (72, 180)       | (14 × 7)     | 0     |                               |
| **Level Display** | (72, 180)       | (14 × 7)     | 1     | Shown only outside bonus phase   |









