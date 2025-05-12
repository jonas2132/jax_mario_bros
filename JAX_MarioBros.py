import pygame
import chex
import jax
import jax.numpy as jnp
from jax import jit

# --- Konstanten ---
RENDER_SCALE = 3
WIDTH, HEIGHT = 160, 210
GRAVITY = 0.5
JUMP_VELOCITY = -10.0
MOVE_SPEED = 3.0
PLAYER_SIZE = (30, 30)
PLATFORM_RECT = (0, 500, WIDTH * RENDER_SCALE, 20)  # x, y, w, h

# --- GameState mit chex ---
@chex.dataclass
class GameState:
    pos: jnp.ndarray     # [x, y]
    vel: jnp.ndarray     # [vx, vy]
    on_ground: bool

# --- Initialzustand ---
def init_state():
    return GameState(pos=jnp.array([200.0, 300.0]),
                     vel=jnp.array([0.0, 0.0]),
                     on_ground=False)

# --- Plattform-Kollision ---
def check_collision(pos: jnp.ndarray):
    x, y = pos
    w, h = PLAYER_SIZE
    px, py, pw, ph = PLATFORM_RECT

    # Spieler-Rechteck
    player_left = x
    player_right = x + w
    player_top = y
    player_bottom = y + h

    # Plattform-Rechteck
    plat_left = px
    plat_right = px + pw
    plat_top = py
    plat_bottom = py + ph

    overlap_x = (player_right > plat_left) & (player_left < plat_right)
    overlap_y = (player_bottom > plat_top) & (player_top < plat_bottom)

    return overlap_x & overlap_y

# --- JIT-kompilierter Step ---
@jit
def step(state: GameState, action: jnp.ndarray) -> GameState:
    move = action[0]  # -1 = links, 0 = nichts, 1 = rechts
    jump = action[1]  # 1 = springen

    vx = MOVE_SPEED * move
    vy = state.vel[1] + GRAVITY
    vy = jnp.where(jump & state.on_ground, JUMP_VELOCITY, vy)

    new_pos = state.pos + jnp.array([vx, vy])
    collided = check_collision(new_pos)
    falling = vy > 0  # Nur "landen", wenn wir nach unten fallen
    landed = collided & falling

    corrected_y = jnp.where(landed, PLATFORM_RECT[1] - PLAYER_SIZE[1], new_pos[1])
    new_vy = jnp.where(landed, 0.0, vy)

    return GameState(
        pos=jnp.array([new_pos[0], corrected_y]),
        vel=jnp.array([vx, new_vy]),
        on_ground=collided
    )

# --- Hauptprogramm ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH*RENDER_SCALE, HEIGHT * RENDER_SCALE))
    pygame.display.set_caption("JAX_MarioBros")
    clock = pygame.time.Clock()

    state = init_state()
    running = True

    while running:
        keys = pygame.key.get_pressed()
        move = -1 if keys[pygame.K_LEFT] else (1 if keys[pygame.K_RIGHT] else 0)
        jump = 1 if keys[pygame.K_SPACE] else 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = jnp.array([move, jump], dtype=jnp.int32)
        state = step(state, action)

        # --- Zeichnen ---
        screen.fill((0, 0, 0))
        pygame.draw.rect(screen, (255, 0, 0), (*state.pos.tolist(), *PLAYER_SIZE))
        pygame.draw.rect(screen, (100, 100, 100), PLATFORM_RECT)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
