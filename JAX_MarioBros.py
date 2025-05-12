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
GRAVITY = 0.5
JUMP_VELOCITY = -8.0
MOVE_SPEED = 2.0

# --- Spieler-Größe ---
PLAYER_SIZE = (9, 21)  # Breite, Höhe

# --- Plattformen: Liste von (x, y, w, h) ---
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

# --- Initialzustand ---
def init_state():
    # Startposition über zentraler Ebene
    return GameState(
        pos=jnp.array([37.0, 74.0]),   # 37,74 passt zur mittleren Plattform
        vel=jnp.array([0.0, 0.0]),
        on_ground=False
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

    # POW-Kollision (nur von unten)
    pow_x, pow_y, pow_w, pow_h = pow_block[0]
    pow_left, pow_right = pow_x, pow_x + pow_w
    pow_top, pow_bottom = pow_y, pow_y + pow_h

    pow_overlap_x = (right > pow_left) & (left < pow_right)
    pow_hit_from_below = pow_overlap_x & (vy < 0) & (top - vy >= pow_bottom) & (top <= pow_bottom)
    pow_bump_y = jnp.where(pow_hit_from_below, pow_bottom, -jnp.inf)

    pow_bumped = pow_hit_from_below
    pow_y_new = jnp.max(pow_bump_y)

    return jnp.any(landed), jnp.any(bumped | pow_bumped), new_y_land, jnp.maximum(new_y_bump, pow_y_new), pow_bumped




# --- JIT-kompilierte Schritt-Funktion ---
@jit
def step(state: GameState, action: jnp.ndarray) -> GameState:
    move, jump = action[0], action[1]

    vx = MOVE_SPEED * move
    vy = state.vel[1] + GRAVITY
    vy = jnp.where((jump == 1) & state.on_ground, JUMP_VELOCITY, vy)

    new_pos = state.pos + jnp.array([vx, vy])

    landed, bumped, y_land, y_bump, pow_hit = check_collision(new_pos, jnp.array([vx, vy]), PLATFORMS, POW_BLOCK)

    new_y = jnp.where(landed, y_land,
                      jnp.where(bumped, y_bump, new_pos[1]))
    new_vy = jnp.where(landed | bumped, 0.0, vy)
    new_x = jnp.clip(new_pos[0], 0, SCREEN_WIDTH - PLAYER_SIZE[0])

    return GameState(
        pos=jnp.array([new_x, new_y]),
        vel=jnp.array([vx, new_vy]),
        on_ground=landed
    )





# --- Hauptprogramm mit pygame ---
def main():
    pygame.init()
    screen = pygame.display.set_mode(
        (SCREEN_WIDTH * WINDOW_SCALE, SCREEN_HEIGHT * WINDOW_SCALE)
    )
    pygame.display.set_caption("JAX Mario Bros Prototype")
    clock = pygame.time.Clock()

    state = init_state()
    running = True

    # Hilfsfunktion zum Skalieren beim Zeichnen
    def draw_rect(color, rect):
        r = pygame.Rect(rect)
        r.x *= WINDOW_SCALE; r.y *= WINDOW_SCALE
        r.w *= WINDOW_SCALE; r.h *= WINDOW_SCALE
        pygame.draw.rect(screen, color, r)

    while running:
        # Input
        keys = pygame.key.get_pressed()
        move = -1 if keys[pygame.K_LEFT] else (1 if keys[pygame.K_RIGHT] else 0)
        jump = 1 if keys[pygame.K_SPACE] else 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Game-Logic
        action = jnp.array([move, jump], dtype=jnp.int32)
        state = step(state, action)

        # Rendering
        screen.fill((0, 0, 0))
        # Spieler
        draw_rect((181,  83,  40), (*state.pos.tolist(), *PLAYER_SIZE))
        # Plattformen
        for plat in PLATFORMS:
            draw_rect((228, 111, 111), plat.tolist())

        draw_rect((201, 164, 74), POW_BLOCK[0].tolist())
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
