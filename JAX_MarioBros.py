import sys
import pygame
import chex
import jax
import jax.numpy as jnp
from jax import jit
from dataclasses import dataclass

# --- Bildschirmgröße ---
SCREEN_WIDTH = 160
SCREEN_HEIGHT = 210
WINDOW_SCALE = 3  # Optional zum Hochskalieren

# --- Physik-Parameter ---
MOVE_SPEED = 1
ASCEND_VY     = -2.0         # ↑ 2 px / frame
DESCEND_VY    =  2.0         # ↓ 2 px / frame
ASCEND_FRAMES = 21         # 42 px tall jump (21 × 2)

# --- Spieler-und-Enemy-Größe ---
PLAYER_SIZE = (9, 21)  # w, h
PLAYER_COLOR = (181, 83, 40)
ENEMY_SIZE = (8, 8)  # w, h

# --- Coin parameters ---
COIN_SIZE = 6
COIN_COLOR = (255, 215, 0)  # Gold
COIN_SCORE = 800
COIN_STAGE_DURATION = 15 * 30  # 15 seconds at 30 FPS
COIN_POSITIONS = jnp.array([
    [35.0, 55.0], [120.0, 55.0],   # Top platforms
    [60.0, 93.0], [100.0, 93.0],   # Middle platform
    [20.0, 133.0], [140.0, 133.0], # Bottom platforms
    [80.0, 55.0]                    # Center top
])

PLATFORMS = jnp.array([
    [0, 168, 160, 24],   # Boden
    [0, 57, 64, 3],   # Plattform 1
    [96, 57, 68, 3],  # Plattform 2
    [31, 95, 97, 3],  # Plattform 3 (hier könnte der Pow Block sein)
    [0, 95, 16, 3],   # Plattform 4
    [144, 95, 18, 3], # Plattform 5
    [0, 135, 48, 3],  # Plattform 6
    [112, 135, 48, 3] # Plattform 7
])

# --- Pow_Block ---
POW_BLOCK = jnp.array([[72, 135, 16, 7]])  # x, y, w, h


# --- GameState mit chex ---
@chex.dataclass
class GameState:
    pos: jnp.ndarray     # [x, y]
    vel: jnp.ndarray     # [vx, vy]
    on_ground: bool
    jump_phase: int
    ascend_frames: int
    enemy_pos: jnp.ndarray  # shape (N, 2)
    enemy_vel: jnp.ndarray  # shape (N, 2)
    game_phase: jnp.int32   # 0=normal, 1=coin_stage
    coin_timer: jnp.int32
    coins_active: jnp.ndarray  # boolean array
    score: jnp.int32
    kills: jnp.int32          # Track enemy kills

# --- Initialzustand ---
def init_state():
    enemy_pos = jnp.array([[50.0, 160.0], [120.0, 160.0]])
    enemy_vel = jnp.array([[0.5, 0.0], [-0.5, 0.0]])
    # Startposition über zentraler Ebene
    return GameState(
        pos=jnp.array([37.0, 74.0]),   # 37,74 passt zur mittleren Plattform
        vel=jnp.array([0.0, 0.0]),
        on_ground=False,
        jump_phase=jnp.int32(0),
        ascend_frames=jnp.int32(0),
        enemy_pos=enemy_pos,
        enemy_vel=enemy_vel,
        game_phase=jnp.int32(0),      # Start in normal phase
        coin_timer=jnp.int32(0),
        coins_active=jnp.zeros(COIN_POSITIONS.shape[0], dtype=bool),
        score=jnp.int32(0),
        kills=jnp.int32(0)

    )

# --- AABB-Kollision: Boden (landed) und Decke (bumped) ---
def check_collision(pos: jnp.ndarray, vel: jnp.ndarray, platforms: jnp.ndarray, pow_block: jnp.ndarray):
    x, y = pos
    vx, vy = vel
    w, h = PLAYER_SIZE

    left, right = x, x + w
    top, bottom = y, y + h

    # Plattformen
    px, py, pw, ph = platforms[:, 0], platforms[:, 1], platforms[:, 2], platforms[:, 3]
    p_left, p_right = px, px + pw
    p_top, p_bottom = py, py + ph

    overlap_x = (right > p_left) & (left < p_right)
    overlap_y = (bottom > p_top) & (top < p_bottom)
    collided = overlap_x & overlap_y

    landed = collided & (vy > 0) & (bottom - vy <= p_top)
    bumped = collided & (vy < 0) & (top - vy >= p_bottom)

    # Höhenkorrektur
    landing_y = jnp.where(landed, p_top - h, jnp.inf)
    bumping_y = jnp.where(bumped, p_bottom, -jnp.inf)
    new_y_land = jnp.min(landing_y)
    new_y_bump = jnp.max(bumping_y)

    # POW block (only bump from below)
    pow_x, pow_y, pow_w, pow_h = pow_block[0]
    pow_left, pow_right = pow_x, pow_x + pow_w
    pow_top, pow_bottom = pow_y, pow_y + pow_h

    pow_overlap_x = (right > pow_left) & (left < pow_right)
    pow_hit_from_below = pow_overlap_x & (vy < 0) & (top - vy >= pow_bottom) & (top <= pow_bottom)
    pow_bump_y = jnp.where(pow_hit_from_below, pow_bottom, -jnp.inf)

    pow_bumped = pow_hit_from_below
    pow_y_new = jnp.max(pow_bump_y)

    return jnp.any(landed), jnp.any(bumped | pow_bumped), new_y_land, jnp.maximum(new_y_bump, pow_y_new), pow_bumped

def check_enemy_collision(player_pos, enemy_pos):
    px, py = player_pos
    pw, ph = PLAYER_SIZE
    ex, ey = enemy_pos[:, 0], enemy_pos[:, 1]
    ew, eh = ENEMY_SIZE

    overlap_x = (px < ex + ew) & (px + pw > ex)
    overlap_y = (py < ey + eh) & (py + ph > ey)
    return jnp.any(overlap_x & overlap_y)


# --- JIT-kompilierte Schritt-Funktion ---
@jit
def step(state: GameState, action: jnp.ndarray) -> GameState:
    move, jump_btn = action[0], action[1].astype(jnp.int32)
    vx = MOVE_SPEED * move
    # -------- phase / frame bookkeeping --------------------------
    start_jump = (jump_btn == 1) & state.on_ground & (state.jump_phase == 0)

    jump_phase = jnp.where(start_jump, 1, state.jump_phase)
    asc_left   = jnp.where(start_jump, ASCEND_FRAMES, state.ascend_frames)

    # vertical speed for this frame
    vy = jnp.where(
            jump_phase == 1, ASCEND_VY,
            jnp.where(jump_phase == 2, DESCEND_VY,
                      jnp.where(state.on_ground, 0.0, DESCEND_VY))
         )

    # integrate position
    new_pos = state.pos + jnp.array([vx, vy])

    landed, bumped, y_land, y_bump, pow_hit = check_collision(new_pos, jnp.array([vx, vy]), PLATFORMS, POW_BLOCK)

    new_y = jnp.where(landed, y_land,
              jnp.where(bumped, y_bump, new_pos[1]))

    # ---------- update phases after collision & time -------------
    # decrement ascend frames while ascending
    asc_left = jnp.where(jump_phase == 1, jnp.maximum(asc_left - 1, 0), asc_left)
    # switch to descend when ascent finished
    jump_phase = jnp.where((jump_phase == 1) & (asc_left == 0), 2, jump_phase)
    # head bump → descend immediately
    jump_phase = jnp.where(bumped & (vy < 0), 2, jump_phase)
    asc_left   = jnp.where(bumped & (vy < 0), 0, asc_left)
    # landing → reset
    jump_phase = jnp.where(landed, 0, jump_phase)
    asc_left   = jnp.where(landed, 0, asc_left)
    # walked off ledge → fall
    jump_phase = jnp.where((jump_phase == 0) & (~landed), 2, jump_phase)

    vy_final = jnp.where(
        jump_phase == 1, ASCEND_VY,
        jnp.where(jump_phase == 2, DESCEND_VY, 0.0)
    )

    new_x = jnp.clip(new_pos[0], 0, SCREEN_WIDTH - PLAYER_SIZE[0])

    return GameState(
        pos=jnp.array([new_x, new_y]),
        vel=jnp.array([vx, vy_final]),
        on_ground=landed,
        jump_phase=jump_phase.astype(jnp.int32),
        ascend_frames=asc_left.astype(jnp.int32),
        enemy_pos=state.enemy_pos,
        enemy_vel=state.enemy_vel,
        game_phase=state.game_phase,
        coin_timer=state.coin_timer,
        coins_active=state.coins_active,
        score=state.score,
        kills=state.kills
    )

# -------------------- MAIN ----------------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode(
        (SCREEN_WIDTH * WINDOW_SCALE, SCREEN_HEIGHT * WINDOW_SCALE)
    )
    pygame.display.set_caption("JAX Mario Bros Prototype")
    clock = pygame.time.Clock()

    # initiwlize font for score and timer
    font = pygame.font.SysFont(None, 24 * WINDOW_SCALE)

    state = init_state()
    running = True

    # -------- pattern management ----------------------------------
    movement_pattern        = [1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0]
    pat_len        = len(movement_pattern)
    idx_right      = 0      # current index if RIGHT is held
    idx_left       = 0      # current index if LEFT is held
    jump = 0
    jumpL = False
    jumpR = False
    # --- New Variables for stop frames -------------------------
    last_dir = 0
    brake_frames_left = 0
    BRAKE_DURATION = 10           # in Frames
    BRAKE_TOTAL_DISTANCE = 7.0    # in Pixels
    brake_speed = BRAKE_TOTAL_DISTANCE / BRAKE_DURATION  # ≈ 0.7 px/frame

    def draw_rect(color, rect):
        r = pygame.Rect(rect)
        r.x *= WINDOW_SCALE; r.y *= WINDOW_SCALE
        r.w *= WINDOW_SCALE; r.h *= WINDOW_SCALE
        pygame.draw.rect(screen, color, r)

    while running:
        # ------------------- INPUT --------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_ESCAPE]:     # stops game when Escape is pressed
            running = False

        if state.on_ground:
            move = 0.0
            jump = 0
            jumpL = False
            jumpR = False
            
        if not jumpL and not jumpR:
            pressed_right = keys[pygame.K_RIGHT] and not keys[pygame.K_LEFT]
            pressed_left = keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]
        
        if keys[pygame.K_SPACE]:
            jump = 1
                    
        if jump == 0:
            if pressed_right:
                move = movement_pattern[idx_right]
                idx_right = (idx_right + 1) % pat_len
                idx_left = 0
                last_dir = 1
                brake_frames_left = 0
            elif pressed_left:
                move = -movement_pattern[idx_left]
                idx_left = (idx_left + 1) % pat_len
                idx_right = 0
                last_dir = -1
                brake_frames_left = 0
            else:
                if last_dir != 0 and brake_frames_left == 0 and state.on_ground:
                    brake_frames_left = BRAKE_DURATION

                if brake_frames_left > 0:
                    move = last_dir * brake_speed
                    brake_frames_left -= 1
                    if brake_frames_left == 0:
                        last_dir = 0
                else:
                    move = 0.0
        elif jump == 1:
            if pressed_right or jumpR:
                move = movement_pattern[idx_right]
                idx_right = (idx_right + 1) % pat_len
                idx_left = 0
                jumpR = True
            elif pressed_left or jumpL:
                move = -movement_pattern[idx_left]
                idx_left = (idx_left + 1) % pat_len
                idx_right = 0
                jumpL = True
            brake_frames_left = 0
            last_dir = 0


        # ----------------- UPDATE & RENDER ------------------------
        state = step(state, jnp.array([move, jump], dtype=jnp.float32))

        # Handle enemy collisions
        if state.game_phase == 0:  # Only in normal phase
            collision = check_enemy_collision(state.pos, state.enemy_pos)
            if collision:
                print("Hit by enemy! Respawning...")
                state = init_state().replace(score=state.score)
        
        # Check if we should transition to coin stage
        if state.game_phase == 0 and state.kills >= 2:
            print("Starting coin stage!")
            state = state.replace(
                game_phase=jnp.int32(1),
                coin_timer=jnp.int32(COIN_STAGE_DURATION),
                coins_active=jnp.ones(COIN_POSITIONS.shape[0], dtype=bool),
                enemy_pos=jnp.zeros((0, 2)),  # Remove enemies
                enemy_vel=jnp.zeros((0, 2))
            )
        
        # Handle coin stage logic
        if state.game_phase == 1:  # Coin stage
            # Update timer
            state = state.replace(coin_timer=state.coin_timer - 1)
            
            # Check coin collisions
            new_coins_active = []
            coin_collected = False
            
            for i in range(len(COIN_POSITIONS)):
                if state.coins_active[i]:
                    if check_coin_collision(state.pos, COIN_POSITIONS[i]):
                        new_coins_active.append(False)
                        coin_collected = True
                        state = state.replace(score=state.score + COIN_SCORE)
                        print(f"Coin collected! Score: {state.score}")
                    else:
                        new_coins_active.append(True)
                else:
                    new_coins_active.append(False)
            
            if coin_collected:
                state = state.replace(coins_active=jnp.array(new_coins_active))
            
            # End coin stage conditions
            if state.coin_timer <= 0 or jnp.all(~state.coins_active):
                print("Coin stage ended! Returning to normal gameplay.")
                # Return to normal gameplay with new enemies
                state = init_state().replace(
                    score=state.score,
                    kills=jnp.int32(0)
                )

        screen.fill((0, 0, 0))
        # player
        if brake_frames_left > 0:
            draw_rect((255,0,0), (*state.pos.tolist(), *PLAYER_SIZE))
        else:
            draw_rect(PLAYER_COLOR, (*state.pos.tolist(), *PLAYER_SIZE))
        # platforms
        for plat in PLATFORMS:
            draw_rect((228, 111, 111), plat.tolist())
        # POW block
        draw_rect((201, 164, 74), POW_BLOCK[0].tolist())

        for ep in state.enemy_pos:
            draw_rect((255, 0, 0), (*ep.tolist(), *ENEMY_SIZE))

        # Collision check for enemies (basic death)
        if check_enemy_collision(state.pos, state.enemy_pos):
            print("Hit by enemy! Respawning...")
            state = init_state()
        
        # Draw enemies (only in normal phase)
        if state.game_phase == 0:
            for ep in state.enemy_pos:
                draw_rect((255, 0, 0), (*ep.tolist(), *ENEMY_SIZE))

        # Draw coins (only in coin stage)
        if state.game_phase == 1:
            for i, coin_pos in enumerate(COIN_POSITIONS):
                if state.coins_active[i]:
                    pygame.draw.circle(
                        screen, 
                        COIN_COLOR,
                        (int(coin_pos[0] * WINDOW_SCALE), 
                         int(coin_pos[1] * WINDOW_SCALE)),
                        COIN_SIZE * WINDOW_SCALE
                    )
        
        # Draw score
        score_text = font.render(f"Score: {state.score}", True, (255, 255, 255))
        screen.blit(score_text, (5 * WINDOW_SCALE, 5 * WINDOW_SCALE))
        
        # Draw coin stage timer
        if state.game_phase == 1:
            timer_text = font.render(
                f"Time: {state.coin_timer // 30}", 
                True, 
                (255, 255, 255)
            )
            screen.blit(timer_text, 
                (5 * WINDOW_SCALE, (SCREEN_HEIGHT - 25) * WINDOW_SCALE))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
